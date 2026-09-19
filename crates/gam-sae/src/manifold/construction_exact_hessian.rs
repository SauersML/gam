// [#780] Exact stationarity-Jacobian correction (`apply_exact_hessian_minus_b`),
// the exact inner-fit Hessian apply (`apply_exact_hessian`), and the exact
// stationarity solve (`solve_exact_stationarity`) were extracted verbatim from
// `construction.rs` into this sibling file to keep that file under the #780
// per-file line-count gate. It is `include!`d back into the parent module in
// `construction.rs`, so these methods share that module's scope exactly as
// before (same `impl SaeManifoldTerm`, same `use super::*` imports).

/// The pencil floor for directions of the exact observed information `A = B_raw + ΔC`,
/// in `μ` units (#2673, #2933 F07). ONE predicate for the value path and the gradient
/// path alike (#2080 defect 4, #2253 x #2330). Its owner is gam-solve's
/// [`gam_solve::arrow_schur::exact_a_pencil_floor`], which the reduced exact-A lane reads
/// as well.
///
/// `Φ = Φ(B_raw)` is the positive-definite evidence factor: the conditioned arrow
/// majorizer the inner Newton solve, the IFT solve and the evidence factor are all
/// expressed in. Directions of `A` are the generalized eigenvectors of the pencil
///
/// ```text
///   A w = μ Φ w,     wᵀΦw = 1
/// ```
///
/// so `μ` measures a direction's exact curvature against the problem's own scale. A
/// nonsingular change of coordinates `θ → Lθ` transforms the pencil by congruence,
/// `A → LᵀAL` and `Φ → LᵀΦL`, maps its eigenvectors as `w → L⁻¹w` and leaves every `μ`
/// where it was. The classification, the retained subspace and the pseudo-inverse are
/// therefore properties of the pencil, not of the coordinates it happens to be written
/// in — including nonorthogonal ones.
///
/// The floor is `√ε`: an exact curvature under `√ε` of the majorizer's along the same
/// direction is not resolved in double precision. That is a numerical convention, not a
/// statistical proof that the direction is unidentifiable. A direction under it (a
/// saturated ordered Beta--Bernoulli gate logit has data curvature `∝ σ'(ℓ)² → 0`) is
/// removed from the IFT response `θ̂_ρ = −A⁺g_ρ`, where it would be an unresolved `1/μ`
/// amplification rather than a derivative (the #931 objective↔gradient desync), and
/// `½log|A|` prices it at the majorizer's own curvature
/// ([`SaeManifoldTerm::classify_exact_hessian_basin`]).
///
/// # The rule this replaced (#2933 F07)
///
/// The classification used to diagonalize `A` in Euclidean coordinates and compare each
/// ordinary eigenvalue `λᵢ` with `max(dim·ε·‖A‖₂, √ε·vᵢᵀBvᵢ)`, on the argument that a
/// generalized Rayleigh quotient is congruence-invariant. The quotient of a FIXED vector
/// is, but the ordinary eigenbasis of `LᵀAL` is not the image of the ordinary eigenbasis
/// of `A`, so the rule was not: with `A = diag(1e-10, 1)`, `B = I` and
/// `L = diag(1e5, 1)·Q·diag(1, √2)` (`Q` the 45° rotation) it kept one direction and,
/// after the change of coordinates, none. The absolute floor before it (#2673) failed
/// the same way more plainly. `tests_pencil_classification_2933` pins both congruences.
///
/// # Numerical resolution is a separate floor
///
/// A computed `μᵢ` carries the backward error of the whitening and of the symmetric
/// eigensolver, and that error does depend on the working coordinates. It is its own
/// per-direction floor, [`sae_exact_a_pencil_resolution`], reported separately when it
/// binds: it can pin a direction whose digits were lost, never resolve one.
pub(crate) fn sae_exact_a_pencil_floor() -> f64 {
    gam_solve::arrow_schur::exact_a_pencil_floor()
}

/// The numerical resolution of one computed pencil eigenvalue, in `μ` units (#2933 F07).
/// Its owner is gam-solve's [`gam_solve::arrow_schur::exact_a_pencil_resolution`]:
/// `‖w‖₂²` is where the working coordinates enter, so the same pencil written in
/// worse-conditioned coordinates resolves fewer digits of `μ`, and this floor pins the
/// directions that lost them instead of classifying round-off.
pub(crate) fn sae_exact_a_pencil_resolution(
    dim: usize,
    vector_norm_sq: f64,
    operator_frobenius: f64,
    metric_frobenius: f64,
    curvature: f64,
) -> f64 {
    gam_solve::arrow_schur::exact_a_pencil_resolution(
        dim,
        vector_norm_sq,
        operator_frobenius,
        metric_frobenius,
        curvature,
    )
}

/// The null-band edge on the side of one pencil direction's curvature, for the dense
/// and the matrix-free exact-`A` routes (#2267, #2933 F07), in `μ` units. Its owner is
/// gam-solve's [`gam_solve::arrow_schur::exact_a_band_edge`], which also states the
/// convention a route that eliminates the coordinate block classifies on.
///
/// The price is continuous across the edge only at a full pin, `s = 1`, where the edge is
/// `μ = 1` and `½·ln μ = 0`. A direction crossing a partial pin's edge `s < 1` moves
/// `½log|A|` by `½·ln s`, and one crossing the bare floor by `½·ln √ε ≈ −9`: changes of
/// stratum, which an outer search comparing values across them reads. Job 1163608 measured
/// all three on the reduced lane (`a_band_edge_crossing_moves_the_value_by_the_log_of_its_edge_2933_f07`,
/// jumps in `log|S|`): `1.221e-4 = ln(1 + step)` at `s = 1`, `−0.6930` against
/// `ln 0.5 = −0.6931` at `s = 0.5`, and `−18.0217` against `ln √ε = −18.0218` at `s = 0`.
/// Pool job 598561 (`sae_manifold_euclidean_k2_fit_terminates`) read the bare-floor step. At one ρ, two evaluations
/// whose loss differed by 1.8e-6 priced `½log|A|` at −3.630e3 (all 996 directions
/// retained, the smallest at 1.457× its floor) and at +5.859e2 (480 in band, the largest
/// at 0.614×), and the outer search's halvings compared values from both strata.
pub(crate) fn sae_exact_a_band_edge(
    curvature: f64,
    resolution: f64,
    substituted_stiffness: f64,
) -> f64 {
    gam_solve::arrow_schur::exact_a_band_edge(curvature, resolution, substituted_stiffness)
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

/// Rank-revealing generalized spectral representation of one dense exact-stationarity
/// block (#2933 F07). The materialized operator and its pencil eigensystem remain
/// together so a pseudo-inverse response can be certified against the physical
/// operator that produced it, rather than against a projected Krylov surrogate.
pub(crate) struct ExactHessianSpectralBlock {
    operator: Array2<f64>,
    /// The pencil eigenvalues `μᵢ` of `A w = μ Φ w`, ascending.
    eigenvalues: Array1<f64>,
    /// `W`, SQUARE and `Φ`-orthonormal (`WᵀΦW = I`): every direction of the
    /// materialized operator is classified by `rank_floor` and nothing is deleted
    /// ahead of it. #2674 — the analytic chart orbit used to be quotiented out here
    /// before diagonalization, which deleted directions the penalized operator has
    /// genuine curvature and genuine slope in.
    eigenvectors: Array2<f64>,
    /// `wᵢᵀ(Φ − B_raw)wᵢ` for every direction: the part of its unit metric the
    /// evidence factor substituted rather than measured. It sets the positive band
    /// edge; see [`sae_exact_a_band_edge`] (#2267).
    substituted_stiffness: Array1<f64>,
    /// The numerical resolution of every computed `μᵢ`; see
    /// [`sae_exact_a_pencil_resolution`].
    resolution: Array1<f64>,
    /// `log|Φ|`. Every in-band direction is priced at the metric's own curvature, so
    /// the metric's determinant is part of the priced `log|A|`.
    metric_log_det: f64,
    /// `‖A‖_F` and `‖Φ‖_F`, the scales the numerical resolution and the certificates
    /// are denominated in.
    operator_frobenius: f64,
    metric_frobenius: f64,
    /// The in-band directions `Z` (ascending) and their metric images `ΦW_Z`, one
    /// column per entry of `band`: the dual components a pseudo-inverse removes from a
    /// right-hand side.
    band: Vec<usize>,
    band_metric_images: Array2<f64>,
    /// #2234 — present when this block prices the orbit-stiffened `A_s` of a dense evaluation:
    /// the exact-`A` pseudo-inverse through the eliminated orbit coordinate, which every solve
    /// and response consumer reads instead of `A_s⁺`.
    orbit: Option<OrbitElimination>,
}

thread_local! {
    /// #2267 — dense exact-`A` pencil decompositions this thread has performed. A dense outer
    /// evaluation decomposes its state once and hands the block to every derivative consumer,
    /// so the count rises by one per evaluated state; the `[SAE-EXACT-DENSE]` line prints it.
    static EXACT_A_PENCIL_DECOMPOSITIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many dense exact-`A` pencil decompositions this thread has performed (#2267).
pub(crate) fn exact_a_pencil_decompositions_on_this_thread() -> u64 {
    EXACT_A_PENCIL_DECOMPOSITIONS.with(std::cell::Cell::get)
}

/// The positive-definite metric an exact-`A` pencil is classified in (#2673,
/// #2933 F07).
///
/// The whitening `L⁻¹AL⁻ᵀ` is built from the metric's own Cholesky factor `Φ = LLᵀ`,
/// applied to a block of right-hand sides at once, so a metric carried as a structured
/// factorization is never materialized as a second `dim × dim` block.
pub(crate) trait ExactAPencilMetric {
    fn dim(&self) -> usize;
    /// `Φv`.
    fn apply(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String>;
    /// `(Φ − B_raw)v`: the stiffness the evidence factor substituted where the
    /// majorizer has no resolved curvature.
    fn substituted_image(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String>;
    /// `L⁻¹V`, one right-hand side per column.
    fn lower_solve(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String>;
    /// `L⁻ᵀV`, one right-hand side per column.
    fn lower_transpose_solve(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String>;
    /// `log|Φ|`.
    fn log_det(&self) -> Result<f64, String>;
    /// `‖Φ‖_F`, read off the metric's own entries rather than `dim` applies (#2267).
    fn frobenius_norm(&self) -> Result<f64, String>;
}

/// The `Φ` metric one spectral block is classified in (#2673).
///
/// `Φ` is the arrow factorization that the inner Newton solve, the IFT solve and the
/// evidence factor are all expressed in, and the ONE thing a direction's curvature is
/// measured against at both the value and the gradient site. Which restriction of it
/// applies is decided by which block of `A` is being classified, so the two cannot be
/// paired up wrongly:
///
/// * the JOINT block is `A` itself, so its metric is the whole arrow operator,
///   border and all.
///
/// Both are applies through the cached factors, never a materialized `Φ`: the dense
/// route already carries one `dim × dim` block and #2724/#2757 price that memory, so a
/// second one would be paid for a scalar per direction. [`Self::prepare`] resolves the
/// metric's block Cholesky factor once.
#[derive(Clone, Copy)]
pub(crate) enum ArrowMetric<'a> {
    /// `Φ` on the joint `(t, β)` coordinates.
    Joint(&'a ArrowFactorCache),
    /// `Φ` on joint `(t, ξ)` coordinates whose border names other variables,
    /// `β = lift·ξ` (#2933 F35): the rank-`r_k` tangent coordinates of learned
    /// Grassmann frames, lifted into the unframed cache's decoder layout. The
    /// pulled-back metric is `diag(I, liftᵀ)·Φ·diag(I, lift)`.
    JointLifted {
        cache: &'a ArrowFactorCache,
        lift: &'a Array2<f64>,
    },
}

impl<'a> ArrowMetric<'a> {
    /// Resolve the metric's block Cholesky factor (#2933 F07). With `T = ⊕ᵢ LᵢLᵢᵀ` the
    /// row factors, `C` the cross block in the metric's border coordinates and `L_b` the
    /// border's Schur factor,
    ///
    /// ```text
    ///   L = [ L_t        0   ]
    ///       [ CᵀL_t⁻ᵀ    L_b ]
    /// ```
    ///
    /// `CᵀL_t⁻ᵀ` is formed here once, from one transpose apply per latent coordinate of the
    /// row cross blocks, so `L⁻¹` and `L⁻ᵀ` on a block of right-hand sides cost row
    /// triangular solves, one product with that block and one border triangular solve. Each
    /// right-hand side used to pay every row's cross-block apply itself, 3·dim times per
    /// whitening (#2933 F36/F39). On the joint layout `L_b` is the cache's reduced-Schur factor
    /// `L_S`. A lift pulls the border back, `C → C·lift`, and the lifted Schur complement is
    /// `liftᵀ·L_S L_Sᵀ·lift`, one `r × r` Gram of `L_Sᵀ·lift` factored here.
    pub(crate) fn prepare(self) -> Result<PreparedArrowMetric<'a>, String> {
        let (cache, lift) = match self {
            Self::Joint(cache) => (cache, None),
            Self::JointLifted { cache, lift } => (cache, Some(lift)),
        };
        let k = cache.k;
        let border_lower = if k == 0 {
            if lift.is_some_and(|lift| lift.ncols() > 0) {
                return Err("ArrowMetric::JointLifted: a lift on a cache with no border".to_string());
            }
            Array2::<f64>::zeros((0, 0))
        } else {
            let Some(schur) = cache.schur_factor.as_ref() else {
                return Err(
                    "ArrowMetric: the pencil metric needs the dense reduced-Schur factor".to_string(),
                );
            };
            if !cache.schur_factor_is_undamped {
                return Err(
                    "ArrowMetric: the Schur factor was not built from the undamped evidence row \
                     factors"
                        .to_string(),
                );
            }
            // The products and blocked solves below read whole blocks, so the factor is carried
            // as its lower triangle alone.
            let schur_lower = Array2::from_shape_fn((k, k), |(row, column)| {
                if row >= column { schur[[row, column]] } else { 0.0 }
            });
            match lift {
                None => schur_lower,
                Some(lift) => {
                    if lift.nrows() != k {
                        return Err(format!(
                            "ArrowMetric::JointLifted: lift has {} rows for a border of {k}",
                            lift.nrows()
                        ));
                    }
                    let factor_image = sequential_transpose_product(&schur_lower, lift);
                    let gram = sequential_transpose_product(&factor_image, &factor_image);
                    gam_linalg::triangular::cholesky_factor_in_place(
                        gram.view(),
                        gam_linalg::triangular::CholeskyGuard::FiniteStrict,
                    )
                    .ok_or_else(|| {
                        "ArrowMetric::JointLifted: the lifted reduced Schur liftᵀ·S·lift is not \
                         positive definite, so the lift is rank deficient in the metric"
                            .to_string()
                    })?
                }
            }
        };
        // Row `i`'s block of `CᵀL_t⁻ᵀ` is `(L_i⁻¹ H_tβ^(i))ᵀ`, and row `c` of `H_tβ^(i)` is
        // `H_βt^(i)e_c`.
        let total_t = cache.delta_t_len();
        let mut cross = Array2::<f64>::zeros((k, total_t));
        if k > 0 {
            let mut image = Array1::<f64>::zeros(k);
            for row in 0..cache.n_rows() {
                let q = cache.row_dims[row];
                let base = cache.row_offsets[row];
                let mut row_cross = Array2::<f64>::zeros((q, k));
                let mut unit = Array1::<f64>::zeros(q);
                for coordinate in 0..q {
                    unit[coordinate] = 1.0;
                    image.fill(0.0);
                    if !cache.apply_htbeta_row_transpose(row, unit.view(), &mut image, None) {
                        return Err(format!("ArrowMetric::prepare: H_βt^({row}) apply failed"));
                    }
                    row_cross.row_mut(coordinate).assign(&image);
                    unit[coordinate] = 0.0;
                }
                let solved = gam_linalg::triangular::forward_substitution_lower_matrix(
                    cache.undamped_factor(row),
                    row_cross.view(),
                );
                cross.slice_mut(s![.., base..base + q]).assign(&solved.t());
            }
        }
        let cross = match lift {
            None => cross,
            Some(lift) => sequential_transpose_product(lift, &cross),
        };
        Ok(PreparedArrowMetric {
            cache,
            lift,
            border_lower,
            cross,
        })
    }
}

/// `AᵀB` by faer's GEMM at [`gam_linalg::faer_ndarray::decomposition_parallelism`], so the
/// whitening's words do not depend on the pool width, as a factorization's do not.
fn sequential_transpose_product<S1, S2>(
    a: &ndarray::ArrayBase<S1, ndarray::Ix2>,
    b: &ndarray::ArrayBase<S2, ndarray::Ix2>,
) -> Array2<f64>
where
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    gam_linalg::faer_ndarray::fast_atb_with_parallelism(
        a,
        b,
        gam_linalg::faer_ndarray::decomposition_parallelism(),
    )
}

/// `L⁻¹B`, or `L⁻ᵀB` when `transpose`, by faer's blocked triangular solve over every
/// right-hand side at once, at the same parallelism as [`sequential_transpose_product`].
/// gam-linalg's `triangular` owner substitutes one right-hand side at a time in a scalar
/// loop, the cost the whitening's border paid per direction (#2933 F36/F39).
fn solve_lower_triangular_block(
    lower: ArrayView2<'_, f64>,
    mut rhs: Array2<f64>,
    transpose: bool,
) -> Array2<f64> {
    let factor = gam_linalg::faer_ndarray::FaerArrayView::new(&lower);
    let solution = gam_linalg::faer_ndarray::array2_to_matmut(&mut rhs);
    let parallelism = gam_linalg::faer_ndarray::decomposition_parallelism();
    if transpose {
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(
            factor.as_ref().transpose(),
            solution,
            parallelism,
        );
    } else {
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            factor.as_ref(),
            solution,
            parallelism,
        );
    }
    rhs
}

/// `Φ` with its border Cholesky factor resolved; see [`ArrowMetric::prepare`].
pub(crate) struct PreparedArrowMetric<'a> {
    cache: &'a ArrowFactorCache,
    lift: Option<&'a Array2<f64>>,
    border_lower: Array2<f64>,
    /// `CᵀL_t⁻ᵀ`, the factor's lower-left block, in the metric's border coordinates.
    cross: Array2<f64>,
}

impl PreparedArrowMetric<'_> {
    fn border_width(&self) -> usize {
        self.lift.map_or(self.cache.k, |lift| lift.ncols())
    }

    fn split_len(&self, len: usize, context: &str) -> Result<usize, String> {
        let total_t = self.cache.delta_t_len();
        if len != total_t + self.border_width() {
            return Err(format!(
                "ArrowMetric::{context}: direction length {len} != joint dimension {}",
                total_t + self.border_width()
            ));
        }
        Ok(total_t)
    }
}

impl ExactAPencilMetric for PreparedArrowMetric<'_> {
    fn dim(&self) -> usize {
        self.cache.delta_t_len() + self.border_width()
    }

    fn apply(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let total_t = self.split_len(v.len(), "apply")?;
        match self.lift {
            None => {
                let b_v = apply_cached_arrow_hessian(
                    self.cache,
                    v.slice(s![..total_t]),
                    v.slice(s![total_t..]),
                )?;
                Ok(Array1::from_iter(b_v.t.iter().chain(b_v.beta.iter()).copied()))
            }
            Some(lift) => {
                let beta = lift.dot(&v.slice(s![total_t..]));
                let b_v = apply_cached_arrow_hessian(self.cache, v.slice(s![..total_t]), beta.view())?;
                let pulled_back = lift.t().dot(&b_v.beta);
                Ok(Array1::from_iter(b_v.t.iter().chain(pulled_back.iter()).copied()))
            }
        }
    }

    /// Read off the row spectra `add_raw_row_deflation_correction` restores `B_raw`
    /// from (#2267): zero on every row the factor kept raw, and on the border, which
    /// that restoration leaves as installed.
    fn substituted_image(&self, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let total_t = self.split_len(v.len(), "substituted_image")?;
        let mut raw_minus_conditioned = Array1::<f64>::zeros(total_t);
        add_raw_row_deflation_correction(
            self.cache,
            v.slice(s![..total_t]),
            raw_minus_conditioned.view_mut(),
            "ArrowMetric::substituted_image",
        )?;
        let mut out = Array1::<f64>::zeros(v.len());
        out.slice_mut(s![..total_t])
            .zip_mut_with(&raw_minus_conditioned, |slot, &value| *slot = -value);
        Ok(out)
    }

    fn lower_solve(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let total_t = self.split_len(v.nrows(), "lower_solve")?;
        let cache = self.cache;
        let mut out = Array2::<f64>::zeros(v.raw_dim());
        for row in 0..cache.n_rows() {
            let rows = cache.row_offsets[row]..cache.row_offsets[row] + cache.row_dims[row];
            out.slice_mut(s![rows.clone(), ..]).assign(
                &gam_linalg::triangular::forward_substitution_lower_matrix(
                    cache.undamped_factor(row),
                    v.slice(s![rows, ..]),
                ),
            );
        }
        if self.border_width() > 0 {
            let coupled = sequential_transpose_product(&self.cross.t(), &out.slice(s![..total_t, ..]));
            let border_rhs = &v.slice(s![total_t.., ..]) - &coupled;
            out.slice_mut(s![total_t.., ..])
                .assign(&solve_lower_triangular_block(self.border_lower.view(), border_rhs, false));
        }
        Ok(out)
    }

    fn lower_transpose_solve(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let total_t = self.split_len(v.nrows(), "lower_transpose_solve")?;
        let cache = self.cache;
        let mut out = Array2::<f64>::zeros(v.raw_dim());
        let mut latent_rhs = v.slice(s![..total_t, ..]).to_owned();
        if self.border_width() > 0 {
            let border = solve_lower_triangular_block(
                self.border_lower.view(),
                v.slice(s![total_t.., ..]).to_owned(),
                true,
            );
            latent_rhs -= &sequential_transpose_product(&self.cross, &border);
            out.slice_mut(s![total_t.., ..]).assign(&border);
        }
        for row in 0..cache.n_rows() {
            let rows = cache.row_offsets[row]..cache.row_offsets[row] + cache.row_dims[row];
            out.slice_mut(s![rows.clone(), ..]).assign(
                &gam_linalg::triangular::back_substitution_lower_transpose_matrix(
                    cache.undamped_factor(row),
                    latent_rhs.slice(s![rows, ..]),
                ),
            );
        }
        Ok(out)
    }

    fn log_det(&self) -> Result<f64, String> {
        let cache = self.cache;
        let mut total = 0.0_f64;
        let mut add_factor = |factor: ArrayView2<'_, f64>, label: &str| -> Result<(), String> {
            for index in 0..factor.nrows() {
                let pivot = factor[[index, index]];
                if !(pivot.is_finite() && pivot > 0.0) {
                    return Err(format!(
                        "ArrowMetric::log_det: {label} pivot {index} is {pivot:e}, not a positive \
                         Cholesky diagonal"
                    ));
                }
                total += 2.0 * pivot.ln();
            }
            Ok(())
        };
        for row in 0..cache.n_rows() {
            add_factor(cache.undamped_factor(row), "row factor")?;
        }
        add_factor(self.border_lower.view(), "border Schur")?;
        Ok(total)
    }

    fn frobenius_norm(&self) -> Result<f64, String> {
        cached_arrow_hessian_frobenius(self.cache, self.lift)
    }
}

/// One point of the Levenberg--Marquardt path of the LINEAR residual model,
/// read off the eigensystem that is already materialized.
///
/// For damping `ν ≥ 0` the step
///
/// ```text
///   Δ(ν) = Σ_i w_i μ_i (w_iᵀ rhs) / (μ_i² + ν)
/// ```
///
/// is the exact minimizer of `‖rhs − AΔ‖²_{Φ⁻¹} + ν‖Δ‖²_Φ`, and `ν = 0` reproduces the
/// pseudoinverse step of [`ExactHessianSpectralBlock::solve_stationarity`]
/// (same `rank_floor`, same retained band). Because the eigensystem is already
/// in hand, the WHOLE path costs one diagonal pass per point — no
/// refactorization, no second operator apply.
///
/// The caller prices the trial point in the currency its convergence gate
/// owns (#2080: the penalized objective); this lower-level spectral object
/// deliberately does not attach an ambient or quotient scalar merit to it.
pub(crate) struct DampedResidualStep {
    /// `Δ(ν)`.
    pub(crate) step: SaeArrowVector,
    /// `‖Δ(ν)‖²_Φ`, in which the retained components are orthonormal.
    pub(crate) step_norm_sq: f64,
    /// Directions whose damped denominator cleared the null band.
    pub(crate) retained_rank: usize,
    /// `‖g‖²_{Φ⁻¹}` carried by the directions inside the null band. The step moves
    /// nothing along them, so this is the part of the stationarity residual that
    /// no step of this operator can reduce, at any damping.
    pub(crate) excluded_gradient_norm_sq: f64,
}

impl ExactHessianSpectralBlock {
    /// The null-band half-width for pencil direction `index`, in `μ` units
    /// (#2673, #2933 F07): [`sae_exact_a_band_edge`] on the direction's own
    /// resolution and substituted stiffness.
    ///
    /// The pencil floor `√ε` is the term that decides the classification: a direction
    /// under it is one whose `A⁺` response would be an unresolved `1/μ` amplification,
    /// so the value must not price a `ρ`-dependence there that the adjoint has projected
    /// out. The resolution term is not a second classification. It pins directions whose
    /// computed `μ` has no significant digits in the working coordinates, and it can only
    /// pin, never resurrect a direction the pencil floor has deflated.
    ///
    /// #2267 — on the positive side the edge rises to the stiffness the evidence
    /// factor substituted along the direction, where that exceeds the floor. The value,
    /// the differential, the polish and the solves all read the band through this one
    /// function, so they move together.
    fn rank_floor(&self, index: usize) -> f64 {
        sae_exact_a_band_edge(
            self.eigenvalues[index],
            self.resolution[index],
            self.substituted_stiffness[index],
        )
    }

    /// Directions discarded because their computed `μ` has no resolved digits at its own
    /// scale although `|μ|` clears the pencil floor. Both adjoint routes discard these
    /// directions; they are reported because the working coordinates, not the pencil,
    /// set the rank there.
    fn resolution_band_crossings(&self) -> usize {
        let floor = sae_exact_a_pencil_floor();
        (0..self.eigenvalues.len())
            .filter(|&index| {
                let magnitude = self.eigenvalues[index].abs();
                magnitude > floor && magnitude <= self.resolution[index]
            })
            .count()
    }

    /// Smallest and largest `|μ|` the null band retained, or `None` when the
    /// whole spectrum is inside it. The two set the DERIVED damping ladder the
    /// polish walks: below `μ_min²` a damping cannot change the flattest
    /// resolved direction, and above `μ_max²` it has already flattened every
    /// direction there is, so no ladder needs to leave `[μ_min², μ_max²]`.
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

    /// A spectrally scaled descent step for the scalar objective whose gradient
    /// is `residual`.  This uses `|A|` in the pencil sense, `|μ|` on every retained
    /// direction: every retained component therefore has negative directional
    /// derivative even when the stationarity operator is indefinite.  `nu` has the
    /// same squared-curvature units as the damping ladder.
    fn damped_objective_step(
        &self,
        residual: &SaeArrowVector,
        nu: f64,
    ) -> Result<DampedResidualStep, String> {
        let total_t = residual.t.len();
        let dim = total_t + residual.beta.len();
        if self.eigenvectors.dim() != (dim, dim) || self.eigenvalues.len() != dim {
            return Err("damped objective step: geometry and residual dimensions differ".into());
        }
        if !(nu.is_finite() && nu >= 0.0) {
            return Err(format!(
                "damped objective step: damping must be finite and >= 0; got {nu}"
            ));
        }
        let mut flat = Array1::<f64>::zeros(dim);
        flat.slice_mut(s![..total_t]).assign(&residual.t);
        flat.slice_mut(s![total_t..]).assign(&residual.beta);
        if !flat.iter().all(|value| value.is_finite()) {
            return Err("damped objective step: residual contains a non-finite value".into());
        }
        let coefficients = self.eigenvectors.t().dot(&flat);
        let mut step_coefficients = Array1::<f64>::zeros(dim);
        let mut retained_rank = 0;
        let mut excluded_gradient_norm_sq = 0.0_f64;
        for index in 0..dim {
            let magnitude = self.eigenvalues[index].abs();
            if magnitude > self.rank_floor(index) {
                step_coefficients[index] = -coefficients[index] / (magnitude + nu.sqrt());
                retained_rank += 1;
            } else {
                excluded_gradient_norm_sq += coefficients[index] * coefficients[index];
            }
        }
        let solution = self.eigenvectors.dot(&step_coefficients);
        Ok(DampedResidualStep {
            step: SaeArrowVector {
                t: solution.slice(s![..total_t]).to_owned(),
                beta: solution.slice(s![total_t..]).to_owned(),
            },
            step_norm_sq: step_coefficients.dot(&step_coefficients),
            retained_rank,
            excluded_gradient_norm_sq,
        })
    }

    /// Apply the covariant pseudo-inverse `A⁺ = W_R diag(1/μ_R) W_Rᵀ` (#2933 F07).
    /// Resolved positive and negative modes are both retained; only the pencil null band
    /// `|μ| ≤ rank_floor` is removed, and that band is the ONLY null predicate on this
    /// route (#2674).
    ///
    /// Under `θ → Lθ` a right-hand side maps as `rhs → Lᵀrhs` and the solution as
    /// `x → L⁻¹x`: the Euclidean Moore--Penrose inverse this replaced had no such law. The
    /// solution satisfies `A x = rhs − ΦW_Z W_Zᵀ rhs` and carries no `Φ`-component along the
    /// band. Three independent certificates guard the result: the physical residual on
    /// that projected right-hand side, the dual residual `Wᵀ(A x − P rhs)` on every
    /// direction, and the solution's `Φ`-mass along the band.
    ///
    /// #2228 — a non-empty band is not a failed solve. The result is the step on the
    /// resolvable complement together with the band directions it held out, each with its
    /// `|μ|` and band edge, so a caller that needs the band component reads it there. `Err`
    /// means only that the solve failed a certificate or its input was malformed.
    fn solve_stationarity(&self, rhs: &SaeArrowVector) -> Result<ExactStationaritySolve, String> {
        // #2234 — an orbit-stiffened block prices `A_s`, but the solve belongs to `A`.
        if let Some(orbit) = self.orbit.as_ref() {
            return self.solve_orbit_eliminated_stationarity(orbit, rhs);
        }
        let total_t = rhs.t.len();
        let dim = total_t + rhs.beta.len();
        let spectral_dim = self.eigenvalues.len();
        if self.operator.dim() != (dim, dim)
            || self.eigenvectors.dim() != (dim, spectral_dim)
            || spectral_dim != dim
            || self.band_metric_images.dim() != (dim, self.band.len())
        {
            return Err(format!(
                "dense exact-stationarity pseudoinverse: geometry dimension {:?}, spectrum {}, \
                 eigenvectors {:?}, band images {:?}, but RHS dimension is {dim}",
                self.operator.dim(),
                spectral_dim,
                self.eigenvectors.dim(),
                self.band_metric_images.dim(),
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
        let mut inverse_coefficients = Array1::<f64>::zeros(spectral_dim);
        let mut retained_rank = 0usize;
        for index in 0..spectral_dim {
            let mu = self.eigenvalues[index];
            if mu.abs() > self.rank_floor(index) {
                inverse_coefficients[index] = coefficients[index] / mu;
                retained_rank += 1;
            }
        }
        let solution = self.eigenvectors.dot(&inverse_coefficients);
        let band_coefficients =
            Array1::from_iter(self.band.iter().map(|&index| coefficients[index]));
        let band_removed = self.band_metric_images.dot(&band_coefficients);
        let projected_rhs = &flat_rhs - &band_removed;
        let physical_residual = &self.operator.dot(&solution) - &projected_rhs;
        // Every direction's dual component of the residual vanishes: along the retained
        // range `μᵢdᵢ = cᵢ`, and along the band both sides carry nothing.
        let dual_residual = self.eigenvectors.t().dot(&physical_residual);
        // The solution's `Φ`-components along the band, reprojected from the computed
        // physical vector rather than from the coefficients that built it, so this gate
        // also detects a loss of `Φ`-orthogonality.
        let band_mass = self.band_metric_images.t().dot(&solution);

        let norm = |vector: &Array1<f64>| vector.dot(vector).max(0.0).sqrt();
        let curvature_norm = self
            .eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let solution_norm = norm(&solution);
        let solution_metric_norm = norm(&inverse_coefficients);
        let physical_norm = norm(&physical_residual);
        let dual_norm = norm(&dual_residual);
        let band_mass_norm = norm(&band_mass);
        // Removing the band's dual components rounds at the scale of the right-hand side and
        // of what it removes, so the backward scale carries both even where `P rhs` is small.
        let physical_scale = self.operator_frobenius * solution_norm
            + norm(&flat_rhs)
            + norm(&band_removed)
            + norm(&projected_rhs);
        let dual_scale = curvature_norm * solution_metric_norm + norm(&coefficients);

        let tolerance = f64::EPSILON.sqrt();
        let within = |residual: f64, scale: f64| {
            residual == 0.0 || (scale > 0.0 && residual <= tolerance * scale)
        };
        if !solution.iter().all(|value| value.is_finite())
            || !within(physical_norm, physical_scale)
            || !within(dual_norm, dual_scale)
            || !within(band_mass_norm, solution_metric_norm)
        {
            let resolution_range = self
                .resolution
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(low, high), &value| {
                    (low.min(value), high.max(value))
                });
            return Err(format!(
                "dense exact-stationarity pseudoinverse failed certification: \
                 physical residual {physical_norm:.6e} / backward scale {physical_scale:.6e}, \
                 dual residual {dual_norm:.6e} / scale {dual_scale:.6e}, \
                 band Φ-mass {band_mass_norm:.6e} / solution Φ-norm {solution_metric_norm:.6e}, \
                 tolerance {tolerance:.6e}, rank {retained_rank}/{spectral_dim} on ambient \
                 dimension {dim}, pencil floor {:.6e}, numerical resolution ∈ [{:.6e}, {:.6e}]",
                sae_exact_a_pencil_floor(),
                resolution_range.0,
                resolution_range.1,
            ));
        }

        Ok(ExactStationaritySolve {
            step: SaeArrowVector {
                t: solution.slice(s![..total_t]).to_owned(),
                beta: solution.slice(s![total_t..]).to_owned(),
            },
            band: self
                .band
                .iter()
                .map(|&index| ExactABandDirection {
                    magnitude: self.eigenvalues[index].abs(),
                    edge: self.rank_floor(index),
                })
                .collect(),
            retained_rank,
            negative_curvature: ResolvedNegativeCurvature::of_directions(
                (0..spectral_dim).map(|index| (self.eigenvalues[index], self.rank_floor(index))),
            ),
        })
    }
}

/// The resolved negative curvature a dense stationarity solve retained (#2228): how many
/// directions sit below `−edge`, and the most negative curvature with its band edge.
/// `A⁺` keeps these directions, so its step moves along them toward a saddle of the
/// penalized objective, not toward the mode the inner solve seeks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResolvedNegativeCurvature {
    pub(crate) directions: usize,
    pub(crate) min_curvature: f64,
    pub(crate) edge: f64,
}

impl ResolvedNegativeCurvature {
    /// The resolved negative directions among `(curvature, edge)` pairs, or `None` when
    /// every curvature is at or above its own `−edge`.
    fn of_directions(directions: impl Iterator<Item = (f64, f64)>) -> Option<Self> {
        let mut found: Option<Self> = None;
        for (curvature, edge) in directions.filter(|&(curvature, edge)| curvature < -edge) {
            found = Some(match found {
                Some(found) if found.min_curvature <= curvature => Self {
                    directions: found.directions + 1,
                    ..found
                },
                Some(found) => Self {
                    directions: found.directions + 1,
                    min_curvature: curvature,
                    edge,
                },
                None => Self {
                    directions: 1,
                    min_curvature: curvature,
                    edge,
                },
            });
        }
        found
    }
}

/// One direction the pencil null band held out of a dense stationarity solve
/// (#2933 F07, #2228): its curvature `|μ|` and the band edge it did not clear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactABandDirection {
    pub(crate) magnitude: f64,
    pub(crate) edge: f64,
}

/// A dense exact-stationarity solve (#2228): `A⁺rhs` on the resolvable complement, the
/// band directions it held out, how many directions the complement retained, and the
/// resolved negative curvature among them.
pub(crate) struct ExactStationaritySolve {
    pub(crate) step: SaeArrowVector,
    pub(crate) band: Vec<ExactABandDirection>,
    pub(crate) retained_rank: usize,
    pub(crate) negative_curvature: Option<ResolvedNegativeCurvature>,
}

/// #2228 / #2933 F07 — outcomes of the dense root refinement's pencil solves. Every clone of a
/// term shares one set of counters, so the outer objective's saved-term restore after a value
/// probe cannot erase them; `SaeManifoldOuterObjective::probe_telemetry` reports them.
#[derive(Clone, Debug, Default)]
pub(crate) struct EvidenceRootTelemetry(std::sync::Arc<EvidenceRootCounters>);

#[derive(Debug, Default)]
pub(crate) struct EvidenceRootCounters {
    band_holds: std::sync::atomic::AtomicUsize,
    band_skips: std::sync::atomic::AtomicUsize,
    solve_failures: std::sync::atomic::AtomicUsize,
    negative_curvature_no_steps: std::sync::atomic::AtomicUsize,
    unfactorable_no_steps: std::sync::atomic::AtomicUsize,
    uncertified_refinements: std::sync::atomic::AtomicUsize,
    rounding_floor_stops: std::sync::atomic::AtomicUsize,
    band_refused_commits: std::sync::atomic::AtomicUsize,
}

/// A snapshot of [`EvidenceRootTelemetry`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EvidenceRootCounts {
    /// The band held a direction while the resolvable complement still carried a step.
    pub(crate) band_holds: usize,
    /// The band held every direction, so no root step was taken.
    pub(crate) band_skips: usize,
    /// The geometry or its solve failed, so no root step was taken.
    pub(crate) solve_failures: usize,
    /// The dense pencil resolved a negative curvature, so no root step was taken.
    pub(crate) negative_curvature_no_steps: usize,
    /// The arrow exact-A system does not factor at ridge 0, so it has no exact Newton step and
    /// none was taken.
    pub(crate) unfactorable_no_steps: usize,
    /// A refinement moved the state and recurred, but the refined root did not certify, so
    /// the accepted state was priced.
    pub(crate) uncertified_refinements: usize,
    /// #2822 — the gate sat inside its formation band, so no root step was solved for.
    pub(crate) rounding_floor_stops: usize,
    /// #2822 — a trial the strict contraction would have committed, refused because the two
    /// gates' formation bands overlap.
    pub(crate) band_refused_commits: usize,
}

impl EvidenceRootTelemetry {
    pub(crate) fn counts(&self) -> EvidenceRootCounts {
        use std::sync::atomic::Ordering;
        EvidenceRootCounts {
            band_holds: self.0.band_holds.load(Ordering::Relaxed),
            band_skips: self.0.band_skips.load(Ordering::Relaxed),
            solve_failures: self.0.solve_failures.load(Ordering::Relaxed),
            negative_curvature_no_steps: self
                .0
                .negative_curvature_no_steps
                .load(Ordering::Relaxed),
            unfactorable_no_steps: self.0.unfactorable_no_steps.load(Ordering::Relaxed),
            uncertified_refinements: self.0.uncertified_refinements.load(Ordering::Relaxed),
            rounding_floor_stops: self.0.rounding_floor_stops.load(Ordering::Relaxed),
            band_refused_commits: self.0.band_refused_commits.load(Ordering::Relaxed),
        }
    }
}

/// #2267 — the dense exact-`A` spectral block one evaluation priced `½log|A|` on, with the
/// half of `E = B − A` its pricing reads. The dense criterion produces it once per state and
/// every dense derivative consumer reads it: the rank-charge dispersion's divergence, the
/// priced log-determinant differential and the stationarity adjoint all derive from this one
/// materialized block and classification floor, so an evaluation decomposes `A` once.
pub(crate) struct DenseExactAGeometry {
    block: ExactHessianSpectralBlock,
    /// `E`'s coordinate diagonal, the ARD concave clamp.
    e_diag: Array1<f64>,
    /// `E`'s decoder-prior border block (#2828).
    e_beta: Option<Array2<f64>>,
    total_t: usize,
    /// #2234 — the closure-certified circle orbits `block` was stiffened along, in tangent-column
    /// order: one per orbit alone in its connected block of `A` and `Φ`. The value and its
    /// derivative price the orbits from them, off the same eliminated block.
    orbit_generators: Vec<CircleOrbitGenerator>,
    /// The reconstruction dispersion the value's rank charge priced at this state, which
    /// the gradient's rank-charge derivative reads instead of forming the fitted-response
    /// divergence a second time (#2933 F39). `None` until the value has priced it.
    rank_charge_dispersion: Option<SaeReconstructionDispersion>,
}

/// Value and classified basin spectrum, without realizing a dense differential.
struct ExactHessianBasin {
    log_det: f64,
    negative: Vec<usize>,
    complement: Vec<usize>,
    basis: Array2<f64>,
    rotation: Array2<f64>,
    vectors: Array2<f64>,
    inverse_values: Array1<f64>,
}

/// Differential of one coherently priced spectral block (#2933 F07): the weights the
/// value contracts against `dA`, `dΦ` and `dE` on its rank stratum; see
/// [`SaeManifoldTerm::exact_hessian_basin_differential`].
/// E has a coordinate diagonal and a dense decoder-prior border block.
/// `a_derivative` includes the negative-subspace projector response.
struct ExactHessianPricing {
    a_derivative: Array2<f64>,
    /// The weight on `dΦ`: the in-band projector and the negative subspace's response to
    /// the metric. All zero on a full-rank positive stratum.
    metric_derivative: Array2<f64>,
    clamp_diagonal_derivative: Array1<f64>,
    clamp_border_derivative: Array2<f64>,
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
/// `rank_adjusted_quasi_laplace_complexity` takes `½log_det` (coordinate block included) and
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

include!("exact_stationarity_krylov.rs");

/// #2330 Patch D — shared per-row context for the residual-curvature
/// third-derivative legs: the whitened `√w·M·r` error metric, its `√w` twin,
/// the frozen assignments/jets, the gate family, and the row's error-metric
/// contractions of every atom's decoded curve. One borrow per row replaces the
/// per-call argument tower of the two leg helpers.
#[derive(Clone, Copy)]
struct PatchDResidualCtx<'a> {
    row: usize,
    error_metric: &'a [f64],
    sqrt_w: f64,
    assignments: &'a Array1<f64>,
    second_jets: &'a [Array4<f64>],
    gate: PatchDGate,
    reconstruction: &'a PatchDRowContractions,
}

/// #2933 F01 — how the logits move the gates `a_k` of the reconstruction
/// `f = Σ_k a_k(ℓ)·γ_k(t_k)`, for the same assignment families the row program
/// differentiates.
#[derive(Clone, Copy)]
enum PatchDGate {
    /// `a = softmax(ℓ/τ)`: every free logit moves every gate.
    Softmax { inv_tau: f64 },
    /// `a_k = σ((ℓ_k − θ)/τ)` (ordered Beta–Bernoulli, threshold gate): a logit
    /// moves only its own atom's gate.
    IndependentLogistic { inv_tau: f64 },
    /// TopK mints no free logit.
    Constant,
}

impl PatchDGate {
    fn for_mode(mode: &AssignmentMode) -> Self {
        match *mode {
            AssignmentMode::Softmax { temperature, .. } => Self::Softmax {
                inv_tau: 1.0 / temperature,
            },
            AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => Self::IndependentLogistic {
                inv_tau: 1.0 / temperature,
            },
            AssignmentMode::TopK { .. } => Self::Constant,
        }
    }

    /// `∂a_k/∂ℓ_i`.
    fn first(self, a: &Array1<f64>, k: usize, i: usize) -> f64 {
        let delta = |x: usize, y: usize| if x == y { 1.0 } else { 0.0 };
        match self {
            Self::Softmax { inv_tau } => a[k] * (delta(k, i) - a[i]) * inv_tau,
            Self::IndependentLogistic { inv_tau } => {
                delta(k, i) * a[k] * (1.0 - a[k]) * inv_tau
            }
            Self::Constant => 0.0,
        }
    }

    /// `∂²a_k/∂ℓ_i∂ℓ_j`.
    fn second(self, a: &Array1<f64>, k: usize, i: usize, j: usize) -> f64 {
        let delta = |x: usize, y: usize| if x == y { 1.0 } else { 0.0 };
        match self {
            Self::Softmax { inv_tau } => {
                a[k] * ((delta(k, i) - a[i]) * (delta(k, j) - a[j]) - a[i] * (delta(i, j) - a[j]))
                    * inv_tau
                    * inv_tau
            }
            Self::IndependentLogistic { inv_tau } => {
                delta(k, i) * delta(k, j) * a[k] * (1.0 - a[k]) * (1.0 - 2.0 * a[k])
                    * inv_tau
                    * inv_tau
            }
            Self::Constant => 0.0,
        }
    }

    /// `Σ_k ∂³a_k/∂ℓ_i∂ℓ_j∂ℓ_l · ⟨em, γ_k⟩`.
    ///
    /// For the softmax this contracts the symmetric third derivative
    /// `D³a_k[u,v,w] = a_k/τ³·[ũ_kṽ_kw̃_k − ũ_k C(v,w) − ṽ_k C(u,w) − w̃_k C(u,v) − κ(u,v,w)]`
    /// (`ũ = u − ⟨a,u⟩`, `C(u,v) = Σ a ũṽ`, `κ(u,v,w) = Σ a ũṽw̃`) at the unit
    /// logit directions. Because `Σ_k D³a_k = 0`, the curve contraction may be
    /// centered, `h_k = ⟨em, γ_k⟩ − Σ_j a_j⟨em, γ_j⟩`, and `Σ_k a_k h_k = 0` removes
    /// `κ`, leaving
    /// `τ⁻³·{2a_i a_j a_l(h_i+h_j+h_l) − [i=j]a_i a_l(h_i+h_l) − [i=l]a_i a_j(h_i+h_j)
    ///  − [j=l]a_i a_j(h_i+h_j) + [i=j=l]a_i h_i}`,
    /// which is also the logit derivative of the row program's centered second
    /// moment `τ⁻²·a_j[δ_jl C_j − a_l(C_j + C_l)]`.
    fn third_contraction(
        self,
        a: &Array1<f64>,
        reconstruction: &PatchDRowContractions,
        i: usize,
        j: usize,
        l: usize,
    ) -> f64 {
        match self {
            Self::Softmax { inv_tau } => {
                let h = &reconstruction.centered;
                let mut acc = 2.0 * a[i] * a[j] * a[l] * (h[i] + h[j] + h[l]);
                if i == j {
                    acc -= a[i] * a[l] * (h[i] + h[l]);
                }
                if i == l {
                    acc -= a[i] * a[j] * (h[i] + h[j]);
                }
                if j == l {
                    acc -= a[i] * a[j] * (h[i] + h[j]);
                }
                if i == j && j == l {
                    acc += a[i] * h[i];
                }
                acc * inv_tau * inv_tau * inv_tau
            }
            Self::IndependentLogistic { inv_tau } => {
                if i != j || j != l {
                    return 0.0;
                }
                let s = a[i];
                s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s * s)
                    * inv_tau
                    * inv_tau
                    * inv_tau
                    * reconstruction.value[i]
            }
            Self::Constant => 0.0,
        }
    }
}

/// #2933 F01 — one row's contractions of the error metric `em` against every
/// atom's decoded curve `γ_k = Σ_m B_k[m,·]·φ_m(t_k)` and its coordinate jets,
/// built once per row so each trilinear residual leg is a gate derivative times
/// one stored number instead of a fresh `basis × p` decode.
struct PatchDRowContractions {
    /// `⟨em, γ_k⟩`.
    value: Vec<f64>,
    /// `⟨em, γ_k⟩ − Σ_j a_j·⟨em, γ_j⟩`, the centered component the softmax
    /// moments differentiate (an inactive atom is the zero curve but its gate
    /// still normalizes, exactly as in `execute_softmax_row_program`).
    centered: Vec<f64>,
    /// `⟨em, ∂_x γ_k⟩`, indexed `x`.
    first: Vec<Vec<f64>>,
    /// `⟨em, ∂²_xy γ_k⟩`, indexed `x·d + y`.
    second: Vec<Vec<f64>>,
    /// `⟨em, ∂³_xyz γ_k⟩`, indexed `(x·d + y)·d + z`; identically zero for an
    /// [`AtomThirdJet::CertifiedZero`] basis.
    third: Vec<Vec<f64>>,
}

/// #2933 F02 — one atom's third jet as the exact-A residual leg consumes it.
/// [`SaeManifoldTerm::atom_third_jets`] refuses an evaluator that declares its
/// jet unavailable, so both states here are derivatives the leg contracts
/// exactly.
pub(crate) enum AtomThirdJet {
    /// The evaluator's closed-form `∂³φ`, shaped `(n_obs, basis, d, d, d)`.
    Analytic(ndarray::Array5<f64>),
    /// Every third partial of the basis vanishes identically, so the coord³ leg
    /// is zero without materializing the tensor.
    CertifiedZero,
}

/// #2933 F02 — the capability refusal of an exact observed-information
/// derivative whose atom's evaluator does not expose its third jet. It is not a
/// numerical failure: the leg `⟨Mr, ∂³f⟩` that jet feeds is nonzero in general
/// (`sin t` has third derivative `−cos t`), so there is no value to return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ThirdJetUnavailable {
    pub(crate) atom: String,
}

impl std::fmt::Display for ThirdJetUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "exact observed-information derivative refused: atom '{}' has no analytic or \
             certified-zero basis third jet, so the residual leg <Mr, d3f> cannot be formed",
            self.atom
        )
    }
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
    /// softmax), or a frozen routing whose prior is inert.
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

/// #2731 — the residual-curvature half of `ΔC` at ONE state, contracted once
/// instead of once per apply.
///
/// Legs (1a) and (1b) of [`SaeManifoldTerm::apply_exact_hessian_minus_b_prepared`]
/// are `⟨r_n, ∂²f_ab⟩` and `⟨r_n, ∂²f_aβ⟩`: the metric-applied row residual
/// against the row's second and mixed jets. Both factors belong to the state;
/// only the direction they multiply changes from one apply to the next. The
/// per-apply form rebuilt every row's packed jet buffer
/// (`q·p + q²·p + n_β·p + 2·q·n_β·p` doubles, allocated zeroed) and contracted
/// it again on every probe. On #2731's `p = 2048, charts = 32` cell that buffer
/// is ~24 MB per row, for each of 256 rows, for each of the 290 probes of one
/// dense materialization; job 391502 read 700–827 ms per apply on one core.
///
/// Held per row: `q × q` and `q × n_β` doubles, the order of the `H_tβ` block
/// the factor cache already holds for that row. Empty under a softmax gate, whose
/// resident contracted kernel never materializes the packed channels; that gate
/// carries its row-jet inputs and residual probe rows in `softmax` instead. A
/// plan is valid for exactly the state it was built from; the callers hold
/// `&self` across the applies they share it with, which is the proof.
pub(crate) struct PreparedResidualCurvatureRows {
    rows: Vec<PreparedResidualCurvatureRow>,
    /// `border_channels_for_cache(cache)[β].index`, in border order.
    border_indices: Vec<usize>,
    /// #2822 — the softmax gate's resident row-jet plan; `None` for every other gate.
    softmax: Option<PreparedSoftmaxRowJets>,
    /// #2933 F36 — every embedded-sphere coordinate block of this state, so an
    /// apply sandwiches the raw-ambient legs in the tangent projector `B` was
    /// assembled in. Empty when no atom lives on a sphere.
    sphere_tangents: Vec<SphereTangentBlock>,
}

/// #2822 — the state-only inputs of the softmax residual-curvature HVP
/// (`SaeManifoldTerm::prepare_softmax_row_jets`): the border channels, and per tile of
/// same-shape rows the planner's path, the row-program inputs and the residual probe rows.
pub(crate) struct PreparedSoftmaxRowJets {
    border: Vec<SaeBorderChannel>,
    tiles: Vec<PreparedSoftmaxRowJetTile>,
}

struct PreparedSoftmaxRowJetTile {
    start: usize,
    q: usize,
    path: crate::gpu_kernels::sae_rowjet::SaeRowJetPath,
    inputs: Vec<crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput>,
    probe: Vec<f64>,
    /// The CPU tile's per-state contractions, when the governor admits them.
    bilinear: Option<crate::gpu_kernels::sae_rowjet::bilinear::SaeRowJetBilinearContractions>,
}

struct PreparedResidualCurvatureRow {
    vars: Vec<SaeLocalRowVar>,
    /// `⟨r, ∂²f_ab⟩`, row-major `q × q`.
    residual_tt: Vec<f64>,
    /// `⟨r, ∂²f_aβ⟩`, row-major `q × n_β`.
    residual_tbeta: Vec<f64>,
}

/// #2933 F36 — one embedded-sphere coordinate block of one row: the row-local
/// slot of its first ambient axis and the point it is stored at.
///
/// The arrow assembly writes a sphere block of `B` on its tangent space:
/// `H_tt = P·H·P − ⟨g, t⟩·P + t tᵀ` and `H_tβ = P·H_tβ` with `P = I − t tᵀ`
/// ([`LatentManifold::riemannian_hessian_matrix`]). The residual-curvature legs of
/// `ΔC` contract the raw ambient jets, which carry a radial component off the
/// sphere (`AmbientSphereHarmonicEvaluator`), and the prior legs are written per
/// ambient axis. Added unprojected, they give `A = B + ΔC` a normal row and column
/// that the retraction never moves. Every consumer of `A` would then price a
/// direction that is not a coordinate: `½log|A|`, the IFT solve, the shape
/// covariance, and the fitted-response divergence. The exact information on the
/// sphere is `P·(H + ΔC)·P − ⟨g, t⟩·P`, with the normal pinned at unit curvature
/// as in `B`. So `ΔC` enters sandwiched in the same `P`, and the normal direction
/// keeps exactly the pin.
pub(crate) struct SphereTangentBlock {
    row: usize,
    local: usize,
    point: Vec<f64>,
}

impl SphereTangentBlock {
    /// `P·v` on this block's slots of one row's local vector, `P = I − t tᵀ`, through
    /// `LatentManifold::project_to_tangent`, the projector the assembly converts
    /// `B`'s row with, so the two cannot drift.
    fn project_local(&self, values: &mut ndarray::ArrayViewMut1<'_, f64>) {
        let slots = self.local..self.local + self.point.len();
        let projected = LatentManifold::Sphere {
            dim: self.point.len(),
        }
        .project_to_tangent(
            ndarray::ArrayView1::from(self.point.as_slice()),
            values.slice(s![slots.clone()]),
        );
        values.slice_mut(s![slots]).assign(&projected);
    }
}

/// `P·v` on every sphere block of a joint vector in the cache layout. The blocks
/// name coordinate slots only, so a `(t, β)` vector's border is untouched.
fn project_sphere_tangent_slots(
    blocks: &[SphereTangentBlock],
    row_offsets: &[usize],
    t: &mut ndarray::ArrayViewMut1<'_, f64>,
) {
    for block in blocks {
        let start = row_offsets[block.row];
        let end = row_offsets[block.row + 1];
        block.project_local(&mut t.slice_mut(s![start..end]));
    }
}

impl SaeManifoldTerm {
    /// Every embedded-sphere coordinate block of this state in the row layout
    /// `row_dims`, in row order. The factors come from
    /// [`Self::all_ard_embedded_sphere_factors`], the one walk of the coordinate
    /// manifolds. The slots come from the same [`Self::row_vars_for_row_dim`] map the
    /// assembly's ext-coord manifold is built in, so the projector names the
    /// coordinates `B` projected.
    pub(crate) fn sphere_tangent_blocks(
        &self,
        row_dims: &[usize],
    ) -> Result<Vec<SphereTangentBlock>, String> {
        let spans = self.all_ard_embedded_sphere_factors();
        if spans.iter().all(Vec::is_empty) {
            return Ok(Vec::new());
        }
        if row_dims.len() != self.n_obs() {
            return Err(format!(
                "sphere_tangent_blocks: {} row dimensions for {} observations",
                row_dims.len(),
                self.n_obs()
            ));
        }
        let mut blocks = Vec::new();
        for (row, &q_row) in row_dims.iter().enumerate() {
            let vars = self.row_vars_for_row_dim(row, q_row)?;
            for (local, var) in vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *var else {
                    continue;
                };
                let Some(&(_, dim)) = spans[atom].iter().find(|&&(first, _)| first == axis) else {
                    continue;
                };
                let contiguous = (0..dim).all(|offset| {
                    matches!(
                        vars.get(local + offset),
                        Some(&SaeLocalRowVar::Coord { atom: other, axis: other_axis })
                            if other == atom && other_axis == axis + offset
                    )
                });
                if !contiguous {
                    return Err(format!(
                        "sphere_tangent_blocks: row {row} does not hold atom {atom}'s sphere axes \
                         {axis}..{} in contiguous slots from {local}",
                        axis + dim
                    ));
                }
                let point = self.assignment.coords[atom].row(row);
                blocks.push(SphereTangentBlock {
                    row,
                    local,
                    point: (0..dim).map(|offset| point[axis + offset]).collect(),
                });
            }
        }
        Ok(blocks)
    }

    /// [`Self::sphere_tangent_blocks`] grouped by row: entry `row` lists each sphere
    /// block's `(local start, point)`, empty on a row that holds none.
    pub(crate) fn sphere_tangent_blocks_by_row(
        &self,
        row_dims: &[usize],
    ) -> Result<Vec<Vec<(usize, Vec<f64>)>>, String> {
        let mut by_row = vec![Vec::new(); row_dims.len()];
        for block in self.sphere_tangent_blocks(row_dims)? {
            by_row[block.row].push((block.local, block.point));
        }
        Ok(by_row)
    }
}

/// #2933 F24 — the Riemannian conversion's share of one sphere row's log-determinant
/// θ-derivative.
///
/// On a row holding embedded-sphere blocks `g` at points `x_g`, the factored row is
/// `A_tt = P·M·P − Σ_g c_g·P_g + Σ_g x_g x_gᵀ` and `A_tβ = P·M_tβ`
/// ([`LatentManifold::riemannian_hessian_matrix`] for `B`, [`SphereTangentBlock`] for
/// `ΔC`). Here `P` is the tangent projector on every block and the identity elsewhere,
/// `P_g` the projector of block `g` alone, `c_g = ⟨g_raw, x_g⟩` the normal component
/// of the raw row gradient, and `M`, `M_tβ` the ambient row operator: Gauss–Newton
/// plus prior curvature, plus `ΔC` on the exact operator. A θ direction moves `P`,
/// `c_g` and the pin as well as `M`. Along the ambient slot `i`, with the tower's own
/// ambient derivatives `dM`, `dM_tβ`:
///
/// ```text
/// dA_tt = P dM P + dP M P + P M dP − Σ_g dc_g P_g − c_f dP_f − dP_f
/// dA_tβ = dP M_tβ + P dM_tβ
/// dP_f  = −(e_i x_fᵀ + x_f e_iᵀ) on the block f holding i, 0 otherwise
/// dc_g  = ⟨x_g, A_exact e_i⟩ + g_raw[i]·1[i ∈ g]
/// ```
///
/// `A_exact` is the ambient exact information, the derivative of `g_raw`. The last
/// term of `dA_tt` is the pin's `e xᵀ + x eᵀ`. The tower projects the resulting slot
/// functional by `P`, because the retraction moves a coordinate along its tangent,
/// and each operand above is linear in the direction. Along a border position `β`:
/// `dA_tt = P dM P − Σ_g ⟨x_g, A_exact,tβ e_β⟩ P_g` and `dA_tβ = P dM_tβ`.
pub(crate) struct SphereRowConversion {
    /// `(local start, point)` of each sphere block of the row.
    blocks: Vec<(usize, Vec<f64>)>,
    /// `P`, the tangent projector on every block and the identity elsewhere.
    projector: Array2<f64>,
    /// `P_g`, block `g`'s tangent projector embedded in the row.
    block_projectors: Vec<Array2<f64>>,
    /// Ambient row operator `M`.
    operator: Array2<f64>,
    /// Ambient cross block `M_tβ`, one column per border position.
    border_operator: Array2<f64>,
    /// `x_gᵀ A_exact` over the row slots, per block.
    normal_exact_rows: Vec<Array1<f64>>,
    /// `x_gᵀ A_exact,tβ` over the border positions, per block.
    normal_exact_border: Vec<Array1<f64>>,
    /// `g_raw` over the row slots.
    gradient: Array1<f64>,
    /// `c_g = ⟨g_raw, x_g⟩`, per block.
    weingarten: Vec<f64>,
}

impl SphereRowConversion {
    /// The conversion operands of one row from its jets and `error_metric = √w·M·r`
    /// (see [`SaeManifoldTerm::patchd_row_error_metric`]), built from the same jet
    /// dots the towers differentiate. `None` on a row without a sphere block.
    pub(crate) fn for_row(
        term: &SaeManifoldTerm,
        row: usize,
        blocks: &[(usize, Vec<f64>)],
        jets: &SaeRowJets,
        border_len: usize,
        error_metric: &[f64],
        ard_precisions: &[Array1<f64>],
        ard_axis_periods: &[Vec<Option<f64>>],
        exact_a: bool,
    ) -> Option<Self> {
        if blocks.is_empty() {
            return None;
        }
        let q = jets.vars.len();
        let w_row = term.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
        let mut operator = Array2::<f64>::zeros((q, q));
        let mut exact = Array2::<f64>::zeros((q, q));
        let mut gradient = Array1::<f64>::zeros(q);
        for a in 0..q {
            gradient[a] = sae_dot(jets.first(a), error_metric);
            for b in 0..q {
                let gauss_newton = sae_dot(jets.first(a), jets.first(b));
                let residual = sae_dot(error_metric, jets.second(a, b));
                operator[[a, b]] = gauss_newton + if exact_a { residual } else { 0.0 };
                exact[[a, b]] = gauss_newton + residual;
            }
            if let SaeLocalRowVar::Coord { atom, axis } = jets.vars[a] {
                if !ard_precisions[atom].is_empty() {
                    let prior = ArdAxisPrior::eval(
                        ard_precisions[atom][axis],
                        term.assignment.coords[atom].row(row)[axis],
                        ard_axis_periods[atom][axis],
                    );
                    operator[[a, a]] += w_row
                        * if exact_a {
                            prior.hess
                        } else {
                            prior.psd_majorizer_hess()
                        };
                    exact[[a, a]] += w_row * prior.hess;
                    gradient[a] += w_row * prior.grad;
                }
            }
        }
        let mut border_operator = Array2::<f64>::zeros((q, border_len));
        let mut border_exact = Array2::<f64>::zeros((q, border_len));
        for a in 0..q {
            for beta_pos in 0..border_len {
                let gauss_newton = sae_dot(jets.first(a), jets.beta(beta_pos));
                let residual = sae_dot(error_metric, jets.beta_deriv(a, beta_pos));
                border_operator[[a, beta_pos]] = gauss_newton + if exact_a { residual } else { 0.0 };
                border_exact[[a, beta_pos]] = gauss_newton + residual;
            }
        }
        let mut projector = Array2::<f64>::eye(q);
        let mut block_projectors = Vec::with_capacity(blocks.len());
        let mut normal_exact_rows = Vec::with_capacity(blocks.len());
        let mut normal_exact_border = Vec::with_capacity(blocks.len());
        let mut weingarten = Vec::with_capacity(blocks.len());
        for (local, point) in blocks {
            let dim = point.len();
            let mut block_projector = Array2::<f64>::zeros((q, q));
            let mut row_normal = Array1::<f64>::zeros(q);
            let mut border_normal = Array1::<f64>::zeros(border_len);
            let mut normal_gradient = 0.0;
            // The projector the assembly converts with, `LatentManifold::project_to_tangent`,
            // column by column.
            let sphere = LatentManifold::Sphere { dim };
            let mut unit = Array1::<f64>::zeros(dim);
            for j in 0..dim {
                unit[j] = 1.0;
                let column = sphere.project_to_tangent(
                    ndarray::ArrayView1::from(point.as_slice()),
                    unit.view(),
                );
                unit[j] = 0.0;
                for i in 0..dim {
                    block_projector[[local + i, local + j]] = column[i];
                    projector[[local + i, local + j]] = column[i];
                }
            }
            for i in 0..dim {
                row_normal.scaled_add(point[i], &exact.row(local + i));
                border_normal.scaled_add(point[i], &border_exact.row(local + i));
                normal_gradient += point[i] * gradient[local + i];
            }
            block_projectors.push(block_projector);
            normal_exact_rows.push(row_normal);
            normal_exact_border.push(border_normal);
            weingarten.push(normal_gradient);
        }
        Some(Self {
            blocks: blocks.to_vec(),
            projector,
            block_projectors,
            operator,
            border_operator,
            normal_exact_rows,
            normal_exact_border,
            gradient,
            weingarten,
        })
    }

    /// `(dA_tt, dA_tβ)` along the ambient slot `slot`, from the tower's ambient
    /// derivatives `dm` (`q×q`) and `dm_border` (`q×border`).
    pub(crate) fn slot_derivative(
        &self,
        slot: usize,
        dm: &Array2<f64>,
        dm_border: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut tt = self.projector.dot(dm).dot(&self.projector);
        let mut tbeta = self.projector.dot(dm_border);
        for (g, block_projector) in self.block_projectors.iter().enumerate() {
            tt.scaled_add(-self.normal_exact_rows[g][slot], block_projector);
        }
        if let Some((f, (local, point))) = self
            .blocks
            .iter()
            .enumerate()
            .find(|(_, (local, point))| *local <= slot && slot < *local + point.len())
        {
            let q = self.projector.nrows();
            let mut d_projector = Array2::<f64>::zeros((q, q));
            for i in 0..point.len() {
                d_projector[[slot, local + i]] -= point[i];
                d_projector[[local + i, slot]] -= point[i];
            }
            tt += &d_projector.dot(&self.operator).dot(&self.projector);
            tt += &self.projector.dot(&self.operator).dot(&d_projector);
            tt.scaled_add(-self.gradient[slot], &self.block_projectors[f]);
            tt.scaled_add(-(self.weingarten[f] + 1.0), &d_projector);
            tbeta += &d_projector.dot(&self.border_operator);
        }
        (tt, tbeta)
    }

    /// `(dA_tt, dA_tβ)` along the border position `beta_pos`.
    pub(crate) fn border_derivative(
        &self,
        beta_pos: usize,
        dm: &Array2<f64>,
        dm_border: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut tt = self.projector.dot(dm).dot(&self.projector);
        for (g, block_projector) in self.block_projectors.iter().enumerate() {
            tt.scaled_add(-self.normal_exact_border[g][beta_pos], block_projector);
        }
        (tt, self.projector.dot(dm_border))
    }

    /// `P·v` of a row's slot functional: a coordinate moves along its tangent.
    pub(crate) fn project_slot_functional(&self, values: &Array1<f64>) -> Array1<f64> {
        self.projector.dot(values)
    }
}

impl SaeManifoldTerm {
    /// Contract the residual-curvature legs of `ΔC` at this state. The row
    /// residual and the row program's jets are read exactly as the per-apply
    /// form read them — the same one-row refill, the same `sae_dot` — so an
    /// apply against the plan is bit-identical to one that re-derived them.
    pub(crate) fn prepare_residual_curvature_rows(
        &self,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<PreparedResidualCurvatureRows, String> {
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            return Ok(PreparedResidualCurvatureRows {
                rows: Vec::new(),
                border_indices: Vec::new(),
                softmax: Some(self.prepare_softmax_row_jets(target, cache)?),
                sphere_tangents: self.sphere_tangent_blocks(&cache.row_dims)?,
            });
        }
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let n_border = border.len();
        let row_loss_w = self.row_loss_weights.as_deref();
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        // #2731 — a row's plan reads only that row: its assignments, its row
        // program's jets (a non-softmax refill builds exactly the row it is asked
        // for), and its residual. The rows run on the rayon pool and are gathered in
        // row order, so the plan is bit-identical to the serial loop and the first
        // failing row's error is the one returned. A pool thread holds one row's
        // jets at a time. The dense exact-A build and the materialization forecast
        // each prepare this plan once per polish step.
        use rayon::prelude::*;
        let planned: Vec<Result<PreparedResidualCurvatureRow, String>> = (0..n)
            .into_par_iter()
            .map(|row| -> Result<PreparedResidualCurvatureRow, String> {
                let q = cache.row_dims[row];
                let mut assignments = Array1::<f64>::zeros(k_atoms);
                let a_scratch = assignments.as_slice_mut().ok_or_else(|| {
                    "prepare_residual_curvature_rows: assignment scratch is not contiguous"
                        .to_string()
                })?;
                self.assignment.try_assignments_row_into(row, a_scratch)?;
                // #932 complete schedule: non-softmax gates use their distinct
                // dynamic row program, one row per refill.
                let mut jet_window: std::collections::VecDeque<SaeRowJets> =
                    std::collections::VecDeque::new();
                self.refill_jet_window(row, cache, &second_jets, &border, &mut jet_window)?;
                let jets = jet_window.pop_front().ok_or_else(|| {
                    format!("prepare_residual_curvature_rows: the jet refill built no row {row}")
                })?;
                let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());

                // √w-scaled metric-applied per-row residual `error_metric = √w·M_n r_n`
                // (the SAME object the assembly's β-tier gradient contracts). The
                // data-fit `½ r_nᵀ M_n r_n` has residual curvature `Σ (M_n r_n)·∂²f`,
                // so this is exactly the residual contracted against the raw `∂²f`
                // jets. `M_n = I` on the isotropic path ⇒ `error_metric = √w·r`.
                let mut decoded = vec![0.0_f64; p];
                let mut fitted = Array1::<f64>::zeros(p);
                let mut error = Array1::<f64>::zeros(p);
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

                let mut residual_tt = vec![0.0_f64; q * q];
                for a in 0..q {
                    for b in 0..q {
                        residual_tt[a * q + b] = sae_dot(&error_metric, jets.second(a, b));
                    }
                }
                let mut residual_tbeta = vec![0.0_f64; q * n_border];
                for a in 0..q {
                    for beta_pos in 0..n_border {
                        residual_tbeta[a * n_border + beta_pos] =
                            sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
                    }
                }
                Ok(PreparedResidualCurvatureRow {
                    vars: jets.vars,
                    residual_tt,
                    residual_tbeta,
                })
            })
            .collect();
        let rows = planned
            .into_iter()
            .collect::<Result<Vec<PreparedResidualCurvatureRow>, String>>()?;
        Ok(PreparedResidualCurvatureRows {
            rows,
            border_indices: border.iter().map(|channel| channel.index).collect(),
            softmax: None,
            sphere_tangents: self.sphere_tangent_blocks(&cache.row_dims)?,
        })
    }
}

impl SaeManifoldTerm {
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

    /// Legs (1)–(4) of `Self::apply_exact_hessian_minus_b`. Leg (5) is
    /// `Self::decoder_prior_gap_border_leg`, which
    /// [`Self::apply_exact_hessian_minus_b_prepared`] folds in after these four.
    ///
    /// #2828 — the β leg's plan is a property of the DECODER STATE, not of the
    /// direction, so a caller that applies `ΔC` many times at one state (a dense
    /// materialization's `slots + k` probes, a Krylov solve's iterations) builds
    /// it once and hands it to leg (5) instead of rebuilding it per apply.
    /// Measured on a 10-atom, `p = 16`, `n = 60` fixture with every pair
    /// near-collinear: 2.79 ms of a 15.19 ms apply. None of legs (1)–(4) reads it.
    ///
    /// #2731 — the residual-curvature legs are the same kind of object and take
    /// the same treatment: `residual` is
    /// [`Self::prepare_residual_curvature_rows`] at this state.
    ///
    /// #2933 F36 — on a sphere coordinate block the legs enter as `P·ΔC·P`, the
    /// tangent projector `B` was assembled in (see [`SphereTangentBlock`]). Legs
    /// (4) and (5) live on logit and border slots, where `P` is the identity.
    fn apply_exact_hessian_minus_b_prepared_before_beta_prior_leg(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        if residual.sphere_tangents.is_empty() {
            return self.apply_exact_hessian_minus_b_ambient_legs(rho, cache, v, residual);
        }
        let mut tangent = v.clone();
        project_sphere_tangent_slots(
            &residual.sphere_tangents,
            &cache.row_offsets,
            &mut tangent.t.view_mut(),
        );
        let mut out = self.apply_exact_hessian_minus_b_ambient_legs(rho, cache, &tangent, residual)?;
        project_sphere_tangent_slots(
            &residual.sphere_tangents,
            &cache.row_offsets,
            &mut out.t.view_mut(),
        );
        Ok(out)
    }

    /// Legs (1)–(4) of `ΔC·v` in the ambient coordinates the jets and priors are
    /// written in, before any sphere tangent projection.
    fn apply_exact_hessian_minus_b_ambient_legs(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        self.assignment.validate_rho_domain(rho)?;
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_t = cache.delta_t_len();
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
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
            // #2822 — the resident contracted kernel below never materializes the
            // packed channels. Its row-program inputs, residual probe rows and border
            // channels are state-only, so `residual` carries them for a softmax gate and
            // an apply gathers only the direction.
            let prepared = residual.softmax.as_ref().ok_or_else(|| {
                "apply_exact_hessian_minus_b: a softmax gate's residual plan carries no row-jet plan"
                    .to_string()
            })?;
            let border = &prepared.border;
            // #2304 resident path for the residual-curvature blocks (1a)+(1b):
            // the raw second/mixed jets are contracted on device (when the plan
            // admits it) against the metric-applied √w-scaled residual and the
            // direction's (t, β) coefficients — the packed channel tensors are
            // never materialized. Blocks (2)-(3) below are logit/coord-space
            // prior curvatures with no channel tensors involved and stay on
            // the host.
            {
                let v_t_for_row = |row: usize, q: usize| -> Result<Vec<f64>, String> {
                    let base = cache.row_offsets[row];
                    Ok((0..q).map(|c| v.t[base + c]).collect())
                };
                let v_beta_row: Vec<f64> =
                    border.iter().map(|channel| v.beta[channel.index]).collect();
                let out_ref = &mut out;
                self.contracted_softmax_bilinear_hvp(
                    prepared,
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
        // #932 complete schedule, #2731 contracted once per state: the row
        // program's jets and the row residual were read by
        // `Self::prepare_residual_curvature_rows`; only the direction is read here.
        if residual.rows.len() != n || residual.border_indices.len() != cache.k {
            return Err(format!(
                "apply_exact_hessian_minus_b: residual-curvature plan has {} rows and {} border \
                 channels, but this state has {n} rows and border width {}; a plan is valid only \
                 for the state it was prepared from",
                residual.rows.len(),
                residual.border_indices.len(),
                cache.k,
            ));
        }
        let border_indices = residual.border_indices.as_slice();
        let n_border = border_indices.len();
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let row_plan = &residual.rows[row];
            if row_plan.vars.len() != q {
                return Err(format!(
                    "apply_exact_hessian_minus_b: row {row} was prepared with {} primaries, but \
                     the cache row has {q}",
                    row_plan.vars.len(),
                ));
            }

            // Local t-slice of `v` for this row.
            let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
            if let Some(direction) = ordered_logit_direction.as_mut() {
                for (local, var) in row_plan.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        direction[row * k_atoms + atom] = v_t[local];
                    }
                }
            }

            // (1a) residual curvature, t–t: ΔC_tt[a,b] = ⟨r, ∂²f_ab⟩.
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = row_plan.residual_tt[a * q + b];
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            // (1b) residual curvature, t–β and β–t: ΔC_tβ[a,β] = ⟨r, ∂²f_aβ⟩.
            //      `jets.beta_deriv[a][β]` = ∂(∂f/∂β_β)/∂θ_a (the mixed second jet).
            for a in 0..q {
                for (beta_pos, &index) in border_indices.iter().enumerate() {
                    let r_ab = row_plan.residual_tbeta[a * n_border + beta_pos];
                    // t row picks up β leg of v; β row picks up t leg of v.
                    out.t[base + a] += r_ab * v.beta[index];
                    out.beta[index] += r_ab * v_t[a];
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
            for (a, va) in row_plan.vars.iter().enumerate() {
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
                for (a, va) in row_plan.vars.iter().enumerate() {
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

    /// `ΔC·v` against plans prepared once for this state: legs (1)–(4) from
    /// `Self::apply_exact_hessian_minus_b_prepared_before_beta_prior_leg`, then
    /// leg (5) from `Self::decoder_prior_gap_border_leg`, added to `out.β` in index
    /// order.
    pub(crate) fn apply_exact_hessian_minus_b_prepared(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        let mut out = self.apply_exact_hessian_minus_b_prepared_before_beta_prior_leg(
            rho, cache, v, residual,
        )?;
        if cache.k > 0 {
            let projection = crate::frames::FrameProjection::new(self);
            let leg =
                self.decoder_prior_gap_border_leg(cache, prepared, &projection, v.beta.view())?;
            for (index, &value) in leg.iter().enumerate() {
                out.beta[index] += value;
            }
        }
        Ok(out)
    }

    /// Leg (5) of `ΔC·v` (#2828): the β-tier decoder priors' exact-minus-majorizer
    /// curvature along `v_beta`, in the cache's own border coordinates.
    ///
    /// Until this leg existed `ΔC` had NO β block at all, so `A_ββ` was
    /// whatever PSD majorizer the assembly installed (the repulsion's
    /// Gauss-Newton block, the amplitude barrier's isotropic ridge, the
    /// separation barrier's `|M|` coupling and `lev` ridge) rather than the
    /// second derivative of the objective `assemble_arrow_schur` gradients —
    /// the whole of the #2330 disagreement, and one-directional: a majorizer
    /// only ever OVER-claims curvature, which is exactly how an
    /// `IndefiniteObservedInformation` refusal can fire on a mode that is not
    /// a saddle.
    ///
    /// The remainder is derived in the full-`B` decoder layout because that is
    /// where the priors live and where the assembly writes them (BEFORE the
    /// frame transform). Under an engaged frame the border coordinate is the
    /// factored `C`, and `B = ΦC` with `Φ = blkdiag(I_M ⊗ U_k)`, `U_kᵀU_k = I`,
    /// so the correct factored operator is the congruence `Φᵀ ΔC_ββ Φ` — lift
    /// the direction, apply, project back. That is the same sandwich
    /// `add_factored_repulsion_curvature` applies to the majorizer this
    /// subtracts, so the two stay in one coordinate system.
    ///
    /// `E_ββ = B − A` on the border is this leg's negation, so a caller that keeps
    /// the columns of `k` border probes has the gap border without a second pass.
    fn decoder_prior_gap_border_leg(
        &self,
        cache: &ArrowFactorCache,
        prepared: &PreparedDecoderPriorBetaCurvature,
        projection: &crate::frames::FrameProjection,
        v_beta: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let beta_dim = self.beta_dim();
        let framed = self.last_frames_active && cache.k == self.factored_border_dim();
        if framed {
            let lifted = projection.lift_border_vec(v_beta);
            let delta = self.decoder_prior_exact_minus_majorizer_beta_hvp_prepared(
                prepared,
                lifted.view(),
            )?;
            Ok(projection.project_border_vec(delta.view()))
        } else if cache.k == beta_dim {
            self.decoder_prior_exact_minus_majorizer_beta_hvp_prepared(prepared, v_beta)
        } else {
            Err(format!(
                "apply_exact_hessian_minus_b: border width {} is neither the full-B \
                 beta_dim {beta_dim} nor the factored border dim {}, so the beta-tier \
                 decoder-prior curvature correction has no coordinate system to be \
                 expressed in",
                cache.k,
                self.factored_border_dim(),
            ))
        }
    }
    /// #2828 — every entry `Σ_{r,c} e_beta[r,c]·left[total_t+r, a]·right[total_t+c, b]`,
    /// the border block's contribution to the quadratic forms between the columns of two
    /// bases in the `(t, β)` layout. All zero when the block is absent or the vectors
    /// carry no border rows (the coordinate-only spectral block).
    ///
    /// The block is applied once to each right column, `k²` work per column, and each
    /// entry contracts one left column against that image, `k` work per entry. A scalar
    /// form per entry paid `k²` for each of `left × right` entries: on the high-p
    /// gauge-deflated K=1 circle at p = 512, a repeated eigenvalue run of ~1530 border
    /// directions against k = 1536 put every eu-stack sample after the operator build in
    /// this form (pool job 614715). The image keeps the scalar form's per-row inner sum
    /// and the contraction keeps its row order and its skip of zero left rows, so every
    /// entry is bit-identical to the scalar form.
    fn dropped_curvature_border_forms(
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        left: ndarray::ArrayView2<'_, f64>,
        right: ndarray::ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let mut forms = Array2::<f64>::zeros((left.ncols(), right.ncols()));
        let Some(block) = e_beta else {
            return forms;
        };
        let k = block.nrows();
        if left.nrows() < total_t + k || right.nrows() < total_t + k {
            return forms;
        }
        let mut image = Array2::<f64>::zeros((k, right.ncols()));
        for b in 0..right.ncols() {
            for row in 0..k {
                let mut inner = 0.0_f64;
                for col in 0..k {
                    inner += block[[row, col]] * right[[total_t + col, b]];
                }
                image[[row, b]] = inner;
            }
        }
        for a in 0..left.ncols() {
            for b in 0..right.ncols() {
                let mut acc = 0.0_f64;
                for row in 0..k {
                    let scale = left[[total_t + row, a]];
                    if scale == 0.0 {
                        continue;
                    }
                    acc += scale * image[[row, b]];
                }
                forms[[a, b]] = acc;
            }
        }
        forms
    }

    /// #2336 — the diagonal of `E = B − A` restricted to the ARD periodic
    /// prior's concave-half clamp (block (3) of
    /// `Self::apply_exact_hessian_minus_b`), over the coordinate (t) block; zero
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
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
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
    /// correction `ΔC = A − B` (`Self::apply_exact_hessian_minus_b_prepared`),
    /// against the β-tier plan and the residual-curvature rows prepared once for
    /// this state (#2828, #2731).
    fn apply_exact_hessian_prepared(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        // #2515 — the cache factors the conditioned evidence majorizer
        // `Phi(B_raw)`.  That conditioning is a solve/log-determinant policy, not
        // part of the objective Hessian.  Adding ΔC to it would build
        // `Phi(B_raw) + ΔC` on the dense route while the streaming arrow route
        // builds `Phi(B_raw + ΔC)`.  Recover B_raw first so both routes classify
        // the one statistical operator `A_raw = B_raw + ΔC`.
        let b_v = apply_raw_cached_arrow_hessian(cache, v.t.view(), v.beta.view())?;
        let dc_v = self.apply_exact_hessian_minus_b_prepared(rho, cache, v, prepared, residual)?;
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
    /// `-0.5` contraction) therefore has one dense owner. Ritz vectors are used in
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
            .map(|solve| solve.step)
    }

    /// #2228 / #2933 F07 — the root step a dense pencil solve admits: `−A⁺g` on the
    /// resolvable complement. A band direction belongs to the orbit evidence and the F08
    /// certificate, not to this step, so a non-empty band is counted and logged, never read as
    /// a failed solve. `None` is a skip, counted by kind: the band holds every direction, or
    /// the geometry or its solve failed (an untrustworthy step; committed steps and the
    /// accepted state stand).
    fn evidence_root_step_from_pencil(
        &self,
        solve: Result<ExactStationaritySolve, String>,
    ) -> Option<SaeArrowVector> {
        use std::sync::atomic::Ordering;
        let counters = &self.evidence_root_telemetry.0;
        let solve = match solve {
            Ok(solve) => solve,
            Err(err) => {
                counters.solve_failures.fetch_add(1, Ordering::Relaxed);
                log::info!("[SAE-ROOT] no root step: dense exact-A pseudoinverse: {err}");
                return None;
            }
        };
        // #2228 — `A⁺` retains a resolved negative direction with `1/μ < 0`, so `−A⁺g` would
        // step toward the saddle along it. The root this phase refines is a mode.
        if let Some(negative) = solve.negative_curvature {
            counters.negative_curvature_no_steps.fetch_add(1, Ordering::Relaxed);
            log::info!(
                "[SAE-ROOT] no root step: the pencil resolves {} negative curvature direction(s) \
                 (min μ={:.6e} below −{:.6e})",
                negative.directions,
                negative.min_curvature,
                negative.edge,
            );
            return None;
        }
        // Every edge is at least the pencil floor `√ε`; the ranking cross-multiplies, so no
        // ratio is formed.
        let nearest_edge = solve.band.iter().copied().max_by(|left, right| {
            (left.magnitude * right.edge).total_cmp(&(right.magnitude * left.edge))
        });
        if let Some(nearest) = nearest_edge {
            if solve.retained_rank == 0 {
                counters.band_skips.fetch_add(1, Ordering::Relaxed);
                log::info!(
                    "[SAE-ROOT] no root step: pencil band holds a direction (|μ|={:.6e}, \
                     band={:.6e}); all {} directions are in the band",
                    nearest.magnitude,
                    nearest.edge,
                    solve.band.len(),
                );
                return None;
            }
            counters.band_holds.fetch_add(1, Ordering::Relaxed);
            log::info!(
                "[SAE-ROOT] pencil band holds {} direction(s) (nearest its edge |μ|={:.6e}, \
                 band={:.6e}); stepping on the resolvable complement of rank {}",
                solve.band.len(),
                nearest.magnitude,
                nearest.edge,
                solve.retained_rank,
            );
        }
        Some(SaeArrowVector {
            t: -&solve.step.t,
            beta: -&solve.step.beta,
        })
    }

    /// `Self::apply_exact_hessian_matrix_free` against a β-tier plan prepared
    /// once — the form the Krylov solve installs, so the plan is built once per
    /// solve rather than once per iteration.
    pub(crate) fn apply_exact_hessian_matrix_free_prepared(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        vector: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
        residual: &PreparedResidualCurvatureRows,
    ) -> Result<SaeArrowVector, String> {
        let (base_t, base_beta) =
            matrix_free_arrow_operator_apply(system, cache, vector.t.view(), vector.beta.view())
                .map_err(|error| format!("matrix-free evidence operator: {error}"))?;
        let correction = self.apply_exact_hessian_minus_b_prepared(
            rho, cache, vector, prepared, residual,
        )?;
        let mut out = SaeArrowVector {
            t: &base_t + &correction.t,
            beta: &base_beta + &correction.beta,
        };
        add_raw_row_deflation_correction(
            cache,
            vector.t.view(),
            out.t.view_mut(),
            "apply_exact_hessian_matrix_free",
        )?;
        Ok(out)
    }

    /// Matrix-free exact-stationarity sibling used by the wide-border penalized quasi-Laplace
    /// assignment-strength residual. `system` is the reassembled undamped
    /// bordered operator at the converged inner state; `cache` supplies the same
    /// row factors and H_tbeta operator whose rational log-determinant and shared
    /// inverse-probe bundle were consumed by the value/trace lanes.
    ///
    /// The adjoint uses pencil Ritz pairs of `(A, Φ)` and the covariant projections of
    /// the dense pseudoinverse, with the same per-direction null band (#2933 F07).
    fn solve_exact_stationarity_matrix_free(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        system: &ArrowSchurSystem,
        rhs: &SaeArrowVector,
    ) -> Result<SaeArrowVector, String> {
        // `B` — the CONDITIONED evidence majorizer `Phi(B_raw)`. This is the
        // metric, and it is the right one: it is what `ArrowMetric::Joint` uses
        // on the dense route, and what the criterion's `½log|B|` and the `mu`
        // deflation predicate are denominated in.
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
        // #2828/#2731 — one plan of each kind for the whole Krylov solve, not one
        // per iteration.
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        let apply_a = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            self.apply_exact_hessian_matrix_free_prepared(
                rho, cache, system, vector, &prepared, &residual,
            )
        };
        // #2267 — `B_raw`, the physical majorizer `Phi` conditions. Where they differ
        // the Ritz band edge rises to the stiffness `Phi` substituted, as on the
        // dense route.
        let apply_b_raw = |vector: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            let mut raw = apply_b(vector)?;
            add_raw_row_deflation_correction(
                cache,
                vector.t.view(),
                raw.t.view_mut(),
                "matrix-free raw evidence majorizer",
            )?;
            Ok(raw)
        };
        // Classify pencil Ritz pairs of `(A, Φ)` with the dense route's band and invert on
        // their retained span, exactly as the dense pseudoinverse does.
        solve_exact_stationarity_krylov(rhs, &apply_a, &apply_b, &apply_b_raw)
    }

    /// The raw per-flat-coordinate penalty curvature operators
    /// `M_i = ∂H_raw/∂ρ_i` at a frozen inner state, keyed by flat outer coordinate.
    /// Its consumer is [`Self::exact_stationarity_penalty_derivatives_by_flat`],
    /// the raw exact-`A` derivative map. Each `M_i` is degree-one in
    /// `exp(ρ_i)`: `λ_k·½(S_k+S_kᵀ)⊗I`
    /// on atom `k`'s β-block for smoothing; `w_row·max(α cos κt,0)` on the active
    /// row-local t-slots for periodic ARD (`w_row·α` Euclidean); the softmax
    /// Gershgorin majorizer `w_row·diag(Σ_j|H_kj|)` on the logit slots for the
    /// sparse coordinate, which refuses a compact top-k layout and a non-softmax prior.
    fn raw_penalty_curvature_operators_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let mut operators = DensePenaltyDerivatives::new(cache.delta_t_len() + cache.k);
        self.raw_penalty_curvature_operators_into(rho, cache, &mut operators)?;
        Ok(operators.by_flat)
    }

    /// [`Self::raw_penalty_curvature_operators_by_flat`] written into `sink` (#2234): every
    /// entry sits on a row's coordinate block or on the border block, so an arrow-held weight
    /// contracts it as it is written.
    fn raw_penalty_curvature_operators_into<S: PenaltyDerivativeSink + ?Sized>(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        sink: &mut S,
    ) -> Result<(), String> {
        let total_t = cache.delta_t_len();

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
            let flat = rho
                .kappa_flat_index(a)
                .ok_or_else(|| format!("curvature atom {a} has no flat outer coordinate"))?;
            let atom = self.atoms.get(a).ok_or_else(|| {
                format!(
                    "curvature coordinate names atom {a}, outside term K={}",
                    self.atoms.len()
                )
            })?;
            let Some(ds) = atom.smooth_penalty_kappa_derivative()? else {
                // An atom whose roughness is not curvature-parameterised has no
                // κ to move; leaving the coordinate un-assembled is what makes
                // its gradient entry exactly zero rather than silently wrong.
                continue;
            };
            let m = atom.basis_size();
            let off = beta_offsets[a];
            let r = beta_out_dim(a);
            let lambda = lambda_smooth[a];
            sink.touch(flat);
            for mu in 0..m {
                for nu in 0..m {
                    let ds_sym = 0.5 * (ds[[nu, mu]] + ds[[mu, nu]]);
                    let val = lambda * ds_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        sink.add(flat, total_t + off + nu * r + oc, total_t + off + mu * r + oc, val);
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
            sink.touch(flat);
            for mu in 0..m {
                for nu in 0..m {
                    let s_sym = 0.5 * (s[[nu, mu]] + s[[mu, nu]]);
                    let val = lambda * s_sym;
                    if val == 0.0 {
                        continue;
                    }
                    for oc in 0..r {
                        sink.add(flat, total_t + off + nu * r + oc, total_t + off + mu * r + oc, val);
                    }
                }
            }
        }

        // ARD: C_{k,axis} = w_row·max(α cos κt, 0) (periodic) / w_row·α (Euclidean)
        // on the row-local t-slot for (atom k, axis). An axis on an embedded sphere
        // differentiates the row's Riemannian block instead (#2933 F24, see
        // `ard_sphere_log_precision_derivative`).
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let coord_offsets = self.assignment.coord_offsets();
        let periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        let sphere_factors = self.all_ard_embedded_sphere_factors();
        for row in 0..self.n_obs() {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            let row_atoms: Vec<(usize, usize)> = match self.last_row_layout {
                Some(ref layout) => layout.active_atoms[row]
                    .iter()
                    .copied()
                    .zip(layout.coord_starts[row].iter().copied())
                    .collect(),
                None => (0..self.k_atoms()).map(|kk| (kk, coord_offsets[kk])).collect(),
            };
            for (kk, start) in row_atoms {
                if rho.log_ard[kk].is_empty() {
                    continue;
                }
                let coord = &self.assignment.coords[kk];
                let point = coord.row(row);
                for axis in 0..coord.latent_dim() {
                    let alpha = ard_precisions[kk][axis];
                    let prior = ArdAxisPrior::eval(alpha, point[axis], periods[kk][axis]);
                    let hess = w_row * prior.psd_majorizer_hess();
                    if hess == 0.0 {
                        continue;
                    }
                    let flat = rho.ard_flat_index(kk, axis);
                    sink.touch(flat);
                    match Self::ard_sphere_log_precision_derivative(
                        &sphere_factors[kk],
                        point,
                        axis,
                        start,
                        q,
                        hess,
                        w_row * prior.grad,
                    ) {
                        Some(derivative) => sink.add_row_block(flat, base, &derivative),
                        None => {
                            let g_idx = base + start + axis;
                            sink.add(flat, g_idx, g_idx, hess);
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
                    sink.touch(sparse_flat);
                    for (slot, value) in entries {
                        sink.add(sparse_flat, slot, slot, value);
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

        Ok(())
    }

    /// The ρ-derivative of the EXACT-minus-majorizer
    /// stationarity correction, `∂(ΔC)/∂ρ_i` where `ΔC = A − B`
    /// (`Self::apply_exact_hessian_minus_b`), keyed by flat coordinate. The IFT
    /// sensitivity `∂a/∂ρ_i = A⁺(∂Γ/∂ρ_i − (∂A/∂ρ_i)a)` differentiates the EXACT
    /// stationarity Hessian `A = B + ΔC`, not the majorized solver operator `B = H`
    /// (`raw_penalty_curvature_operators_by_flat` = `∂B/∂ρ`). So the `M_i·a` term must
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
    /// * ThresholdGate sparse: the raw signed logit curvature minus its PSD
    ///   clamp (diagonal), degree-one in `lambda_sparse` — nonzero on exactly
    ///   the logits the gate has switched ON, where the clamp `B` installs is
    ///   a hard zero (#2520).
    /// Smooth is unmajorized (`ΔC` has no smooth part), so its delta is zero and it
    /// is absent from the map. Covered config only (softmax, dense row layout).
    pub(crate) fn exact_stationarity_penalty_derivative_delta_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let mut deltas = DensePenaltyDerivatives::new(cache.delta_t_len() + cache.k);
        self.exact_stationarity_penalty_derivative_delta_into(rho, cache, &mut deltas)?;
        Ok(deltas.by_flat)
    }

    /// [`Self::exact_stationarity_penalty_derivative_delta_by_flat`] written into `sink`
    /// (#2234): every delta is row-local.
    pub(crate) fn exact_stationarity_penalty_derivative_delta_into<S: PenaltyDerivativeSink + ?Sized>(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        sink: &mut S,
    ) -> Result<(), String> {
        let k_atoms = self.k_atoms();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
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
        // #2520 - the ThresholdGate's exact-minus-majorizer remainder, the third
        // member of this map. `B` installs `smooth_psd_clamp(magnitude, 1-2a)`
        // and `dC` restores the raw signed `magnitude*(1-2a)`, so the remainder
        // is nonzero on exactly the logits the gate has switched ON. It is
        // degree-one in `lambda_sparse = e^rho` for the same reason the majorizer
        // is (the clamp is homogeneous in its prefactor and the prefactor carries
        // `lambda_sparse`), so `d(dC)/drho_sparse` IS the remainder itself -- the
        // identical argument the smooth and ARD channels use.
        //
        // Its absence is why the dense exact-A sparse log-det trace disagreed
        // with the finite difference of the value it differentiates IN SIGN on a
        // straddling gate: `dB/drho_sparse` is the clamp, which is a hard `0`
        // there, so the whole of `dA/drho_sparse` on those slots lives here.
        let threshold_gate_remainder: Option<(usize, Array1<f64>)> =
            match (self.assignment.mode, rho.sparse_flat_index()) {
                (AssignmentMode::ThresholdGate { .. }, Some(sparse_flat)) => Some((
                    sparse_flat,
                    crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                        &self.assignment,
                        rho,
                        row_w,
                    )?,
                )),
                _ => None,
            };
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
                sink.touch(sparse_flat);
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
                        sink.add(sparse_flat, base + a, base + b, w_row * delta);
                    }
                }
            }
            // ThresholdGate exact-minus-majorizer remainder on the logit slots.
            if let Some((sparse_flat, remainder)) = threshold_gate_remainder.as_ref() {
                for (a, va) in vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    if atom >= k_atoms {
                        continue;
                    }
                    // Already `w_row`-weighted and fixed-logit-masked by the
                    // shared seam, so no second weighting here.
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        sink.add(*sparse_flat, base + a, base + a, neg);
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
                    sink.add(flat, base + a, base + a, w_row * neg);
                }
            }
        }
        Ok(())
    }

    /// The complete frozen-state derivative of the raw exact stationarity
    /// Hessian, `A_raw = B_raw + ΔC`, keyed by flat outer coordinate.
    ///
    /// Keeping this sum behind one named owner makes it possible to compare the
    /// operator derivative directly with finite differences of
    /// [`Self::materialize_exact_hessian_dense_with_gap_border`], instead of validating only a
    /// downstream trace where spectral classification can obscure which operand
    /// drifted (#2515).
    pub(crate) fn exact_stationarity_penalty_derivatives_by_flat(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<std::collections::BTreeMap<usize, Array2<f64>>, String> {
        let mut derivatives = self.raw_penalty_curvature_operators_by_flat(rho, cache)?;
        for (flat, delta) in self.exact_stationarity_penalty_derivative_delta_by_flat(rho, cache)? {
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
        if (aa.is_some() && ka != ch.atom) || (aw.is_some() && kw != ch.atom) {
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
        let atom = &self.atoms[atom_idx];
        // ∂^{2−l}φ_m over the coord axes.
        let phi = match coord_axes.len() {
            2 => second_jets[atom_idx][[row, m, coord_axes[0], coord_axes[1]]],
            1 => atom.basis_jacobian[[row, m, coord_axes[0]]],
            _ => atom.basis_values[[row, m]],
        };
        let s = assignments[atom_idx];
        let gate_factor = match self.assignment.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                let delta = |i, j| if i == j { 1.0 } else { 0.0 };
                match logit_count {
                    0 => s,
                    1 => {
                        let j = if aa.is_none() { ka } else { kw };
                        s * (delta(atom_idx, j) - assignments[j]) / temperature
                    }
                    _ => s * ((delta(atom_idx, ka) - assignments[ka])
                        * (delta(atom_idx, kw) - assignments[kw])
                        - assignments[ka] * (delta(ka, kw) - assignments[kw]))
                        / (temperature * temperature),
                }
            }
            AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => {
                if ka != atom_idx || kw != atom_idx { return 0.0; }
                match logit_count {
                    0 => s,
                    1 => s * (1.0 - s) / temperature,
                    _ => s * (1.0 - s) * (1.0 - 2.0 * s) / (temperature * temperature),
                }
            }
            AssignmentMode::TopK { .. } => if logit_count == 0 { s } else { 0.0 },
        };
        // eo = Σ_out error_metric[out]·output[out] (the channel's output weighting).
        let p = error_metric.len().min(ch.output.len());
        let mut eo = 0.0_f64;
        for out in 0..p {
            eo += error_metric[out] * ch.output[out];
        }
        sqrt_w * gate_factor * phi * eo
    }

    /// #2933 F01 — the row's [`PatchDRowContractions`]: one decoder contraction
    /// `Σ_out B_k[m,out]·em[out]` per atom, then its dot with each basis jet order.
    /// These read the same basis values, Jacobians and second jets the row program
    /// decodes, so the residual third leg differentiates the `∂²f` that `ΔC` holds.
    fn patchd_row_contractions(
        &self,
        row: usize,
        error_metric: &[f64],
        assignments: &Array1<f64>,
        second_jets: &[Array4<f64>],
        third_jets: &[AtomThirdJet],
    ) -> PatchDRowContractions {
        let k_atoms = self.k_atoms();
        let p = error_metric.len();
        let active_atoms = self
            .last_row_layout
            .as_ref()
            .map(|layout| layout.active_atoms[row].as_slice());
        let mut value = vec![0.0_f64; k_atoms];
        let mut first = Vec::with_capacity(k_atoms);
        let mut second = Vec::with_capacity(k_atoms);
        let mut third = Vec::with_capacity(k_atoms);
        for (k, atom) in self.atoms.iter().enumerate() {
            let d = atom.latent_dim();
            let mut first_k = vec![0.0_f64; d];
            let mut second_k = vec![0.0_f64; d * d];
            let mut third_k = vec![0.0_f64; d * d * d];
            if active_atoms.is_none_or(|active| active.binary_search(&k).is_ok()) {
                let decoder = atom.decoder_coefficients();
                for m in 0..atom.basis_size() {
                    let mut weight = 0.0_f64;
                    for out in 0..p {
                        weight += decoder[[m, out]] * error_metric[out];
                    }
                    value[k] += weight * atom.basis_values[[row, m]];
                    for x in 0..d {
                        first_k[x] += weight * atom.basis_jacobian[[row, m, x]];
                        for y in 0..d {
                            second_k[x * d + y] += weight * second_jets[k][[row, m, x, y]];
                        }
                    }
                    // A certified-zero basis has every third partial identically zero.
                    if let AtomThirdJet::Analytic(jet) = &third_jets[k] {
                        for x in 0..d {
                            for y in 0..d {
                                for z in 0..d {
                                    third_k[(x * d + y) * d + z] += weight * jet[[row, m, x, y, z]];
                                }
                            }
                        }
                    }
                }
            }
            first.push(first_k);
            second.push(second_k);
            third.push(third_k);
        }
        let mean: f64 = (0..k_atoms).map(|k| assignments[k] * value[k]).sum();
        let centered = value.iter().map(|&v| v - mean).collect();
        PatchDRowContractions {
            value,
            centered,
            first,
            second,
            third,
        }
    }

    /// #2330 Patch D — the exact-A residual-curvature THIRD-derivative leg
    /// `⟨error_metric, ∂³f_{a,b,w}⟩`, the second half of `∂ΔC_tt[a,b]/∂θ_w`
    /// (the first half `⟨∂error_metric/∂θ_w, ∂²f⟩ = ⟨jets.first(w), jets.second(a,b)⟩`
    /// is added inline as term 1a). The data fit is `½rᵀMr` so its residual
    /// curvature is `⟨M r, ∂²f⟩`; differentiating the SECOND-jet factor gives this
    /// leg. `error_metric` already carries one `√w·M`; this leg carries the other
    /// `√w`, matching the `⟨error_metric, jets.second⟩` convention exactly.
    ///
    /// #2933 F01 — for `f = Σ_k a_k(ℓ)·γ_k(t_k)` each summand reads one atom's
    /// coordinates, so a triple whose coordinates belong to two atoms is an exact
    /// zero. Every other triple selects ONE term of the product rule for `a_k·γ_k`:
    ///
    /// * `c ≥ 1` coordinates of atom `k` and `3 − c` logits:
    ///   `∂^{3−c}a_k · ⟨em, ∂^c γ_k⟩` (gate-zero/curve-third, gate-first/curve-second,
    ///   gate-second/curve-first);
    /// * three logits `i, j, l`: `Σ_k ∂³a_k/∂ℓ_i∂ℓ_j∂ℓ_l · ⟨em, γ_k⟩`
    ///   (gate-third/curve-zero, [`PatchDGate::third_contraction`]).
    ///
    /// The logits need not share the coordinates' atom: under a softmax gate every
    /// free logit moves every gate. This leg used to return zero for any triple
    /// with mixed atom labels, and for any logit unless the gate was ordered
    /// Beta–Bernoulli, while the exact-A route admits Softmax.
    fn patchd_residual_third_leg(
        &self,
        ctx: &PatchDResidualCtx<'_>,
        a_var: SaeLocalRowVar,
        b_var: SaeLocalRowVar,
        w_var: SaeLocalRowVar,
    ) -> f64 {
        let PatchDResidualCtx {
            sqrt_w,
            assignments,
            gate,
            reconstruction,
            ..
        } = *ctx;
        let mut logits = [0usize; 3];
        let mut n_logits = 0usize;
        let mut axes = [0usize; 3];
        let mut n_axes = 0usize;
        let mut coord_atom: Option<usize> = None;
        for var in [a_var, b_var, w_var] {
            match var {
                SaeLocalRowVar::Logit { atom } => {
                    logits[n_logits] = atom;
                    n_logits += 1;
                }
                SaeLocalRowVar::Coord { atom, axis } => {
                    if coord_atom.is_some_and(|owner| owner != atom) {
                        return 0.0;
                    }
                    coord_atom = Some(atom);
                    axes[n_axes] = axis;
                    n_axes += 1;
                }
            }
        }
        let Some(atom) = coord_atom else {
            return sqrt_w
                * gate.third_contraction(assignments, reconstruction, logits[0], logits[1], logits[2]);
        };
        let d = self.atoms[atom].latent_dim();
        let leg = match n_axes {
            1 => {
                gate.second(assignments, atom, logits[0], logits[1])
                    * reconstruction.first[atom][axes[0]]
            }
            2 => {
                gate.first(assignments, atom, logits[0])
                    * reconstruction.second[atom][axes[0] * d + axes[1]]
            }
            _ => assignments[atom] * reconstruction.third[atom][(axes[0] * d + axes[1]) * d + axes[2]],
        };
        sqrt_w * leg
    }

    pub(crate) fn outer_rho_gradient_ift_rhs(
        &self,
        rho: &SaeManifoldRho,
        j: usize,
        cache: &ArrowFactorCache,
    ) -> Result<SaeArrowVector, String> {
        self.assignment.validate_rho_domain(rho)?;
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let n_params = rho.flat_coordinates().len();
        if j >= n_params {
            return Err(format!(
                "outer_rho_gradient_ift_rhs: coordinate {j} outside rho dim {n_params}"
            ));
        }
        let mut t = Array1::<f64>::zeros(cache.delta_t_len());
        let mut beta = Array1::<f64>::zeros(cache.k);
        if rho.sparse_flat_index() == Some(j) {
            let assignment_grad =
                crate::assignment::assignment_prior_log_strength_target_mixed_weighted(
                    &self.assignment,
                    rho,
                    self.row_loss_weights.as_deref(),
                )?;
            let k_atoms = self.k_atoms();
            let assignment_dim = self.assignment.assignment_coord_dim();
            for row in 0..self.n_obs() {
                let base = cache.row_offsets[row];
                let assignment_base = row * k_atoms;
                match self.last_row_layout {
                    Some(_) => {}
                    None => {
                        for free_idx in 0..assignment_dim {
                            t[base + free_idx] = assignment_grad[assignment_base + free_idx];
                        }
                    }
                }
            }
        } else if (rho.smooth_flat_start()..rho.smooth_flat_start() + rho.log_lambda_smooth.len())
            .contains(&j)
        {
            // #1556: this layout-derived coordinate is one atom's smoothness
            // strength. `∂(penalty)/∂log λ_k = λ_k·S_k C_k` touches ONLY
            // atom `k`'s decoder block; every other atom's RHS is zero.
            let target_atom = j - rho.smooth_flat_start();
            let lambda = rho.lambda_smooth_for(target_atom)?;
            let penalty = self.atoms[target_atom].smooth_penalty();
            self.decoder_penalty_ift_rhs_block(cache, target_atom, lambda, penalty, &mut beta)?;
        } else if let Some(target_atom) = rho
            .kappa_atoms
            .iter()
            .copied()
            .find(|&atom| rho.kappa_flat_index(atom) == Some(j))
        {
            // #2935 — raw sectional curvature enters the inner gradient only through
            // the penalty Gram: `∂g/∂κ_k = λ_k·(½(∂S_k/∂κ + ∂S_k/∂κᵀ) ⊗ I) C_k` on atom
            // `k`'s decoder block. An atom without `∂S/∂κ` has no curvature to move.
            if let Some(ds) = self.atoms[target_atom].smooth_penalty_kappa_derivative()? {
                let lambda = rho.lambda_smooth_for(target_atom)?;
                self.decoder_penalty_ift_rhs_block(cache, target_atom, lambda, ds, &mut beta)?;
            }
        } else {
            // ARD coordinate `j`. `ard_flat_index` maps `(atom, axis)` onto the
            // flat coordinate for both parameterizations; a shared axis is owned
            // by SEVERAL atoms, and the RHS for that one outer coordinate is the
            // SUM of each owning atom's `∂g/∂log α_{atom,axis}` block (chain rule
            // through the broadcast). Those blocks land in disjoint per-atom row
            // slots of `t`, so accumulate every matching atom rather than
            // returning on the first. In `PerAtom` mode exactly one `(atom, axis)`
            // matches, reproducing the historical single-atom RHS.
            let sphere_factors = self.all_ard_embedded_sphere_factors();
            for atom in 0..rho.log_ard.len() {
                for axis in 0..rho.log_ard[atom].len() {
                    if rho.ard_flat_index(atom, axis) != j {
                        continue;
                    }
                    let alpha = ard_precisions[atom][axis];
                    let periods = self.ard_axis_periods(atom);
                    let row_w = self.row_loss_weights.as_deref();
                    let sphere = Self::ard_sphere_factor_containing(&sphere_factors[atom], axis);
                    for row in 0..self.n_obs() {
                        let row_t = self.assignment.coords[atom].row(row);
                        let prior = ArdAxisPrior::eval(alpha, row_t[axis], periods[axis]);
                        // The atom's block start in this row: the dense coordinate
                        // offset, or the compact TopK start. Every sibling caller
                        // adds the axis to that start.
                        let Some(block_start) = sae_coord_penalty_offset(
                            self.last_row_layout.as_ref(),
                            self.assignment.coord_offsets()[atom],
                            row,
                            atom,
                        ) else {
                            continue;
                        };
                        // HT row weighting: this RHS is `∂g/∂log α` of the inner-MAP
                        // stationarity gradient `g`, and the assembly writes that
                        // gradient as `w_row·V'` (full `w_row`, `construction_arrow_schur_assembly.rs`
                        // gt seam). The IFT operator `H` it feeds carries full `w_row`
                        // on this coordinate diagonal (`w·(D_data + prior'')`), so the
                        // RHS must carry the SAME full `w_row` to stay consistent — `V`
                        // is linear in α so `∂(w·V')/∂log α = w·V'`. `None` ⇒ w_row = 1,
                        // bit-for-bit the historical RHS.
                        let w_row = row_w.map_or(1.0, |w| w[row]);
                        let base = cache.row_offsets[row] + block_start;
                        match sphere {
                            // On an embedded unit sphere at `x` the assembled gradient
                            // is its tangent projection `P g`, so the RHS is
                            // `w·V'·P e_a` (#2933 F24).
                            Some((offset, dim)) => {
                                let x = &row_t[offset..offset + dim];
                                for i in 0..dim {
                                    t[base + offset + i] += w_row
                                        * prior.grad
                                        * Self::sphere_tangent_of_axis(x, axis - offset, i);
                                }
                            }
                            None => t[base + axis] += w_row * prior.grad,
                        }
                    }
                }
            }
        }
        Ok(SaeArrowVector { t, beta })
    }

    /// #2231 — the crosscoder block coordinate's IFT RHS
    /// `∂g/∂log λ_ℓ = −½·Jᵀ_M Z̃^{(ℓ)}`, where `g` is the inner stationarity
    /// gradient, `Z̃^{(ℓ)}` is the CURRENTLY-SCALED stacked target masked to
    /// block `ℓ`'s columns, and `Jᵀ_M` is the same metric-whitened,
    /// `√w`-weighted data Jacobian the assembly's `gt = J̃ᵀẽ` uses (the target
    /// enters `g` only through the data residual `r̃ = f − Z̃`, and
    /// `∂Z̃_ℓ/∂log λ_ℓ = ½·Z̃_ℓ`). Feeding this RHS through
    /// `solve_exact_stationarity` gives the block coordinate the SAME
    /// `−½·Γᵀθ̂_ρ` Laplace adjoint every other ρ coordinate carries — without
    /// it the block gradient differentiates a fictitious criterion in which
    /// the fitted state is held fixed (#2087 desync class).
    pub(crate) fn crosscoder_block_ift_rhs(
        &self,
        cache: &ArrowFactorCache,
        target: ArrayView2<'_, f64>,
        col_range: std::ops::Range<usize>,
    ) -> Result<SaeArrowVector, String> {
        let n = self.n_obs();
        let p = self.output_dim();
        if target.nrows() != n || target.ncols() != p {
            return Err(format!(
                "crosscoder_block_ift_rhs: target shape ({}, {}) != ({n}, {p})",
                target.nrows(),
                target.ncols()
            ));
        }
        if col_range.end > p || col_range.start >= col_range.end {
            return Err(format!(
                "crosscoder_block_ift_rhs: block columns {col_range:?} outside output dim {p}"
            ));
        }
        let mut t = Array1::<f64>::zeros(cache.delta_t_len());
        let mut beta = Array1::<f64>::zeros(cache.k);
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let whiten = self.whiten_logdet_row_jets();
        if matches!(self.assignment.mode, AssignmentMode::Softmax { .. }) {
            // #2304 resident path: the packed channel tensors are reduced in
            // place (on device when the plan admits it) and only the per-row
            // t/β coefficients return.
            //
            // The probe is `−½·√w·Z̃` on the block's columns, zero elsewhere
            // (the −½ applied at emit time). With a whitening metric, the
            // historical consumer whitened BOTH the jets and this vector to
            // rank space and dotted there; `⟨Uᵀa, Uᵀv⟩ = ⟨a, U(Uᵀv)⟩`
            // exactly, so the metric folds into the probe as `M_n v` and the
            // raw jets are contracted directly.
            let probe_for_row = |row: usize| -> Result<Vec<f64>, String> {
                let sqrt_w = self
                    .row_loss_weights
                    .as_deref()
                    .map_or(1.0, |w| w[row].sqrt());
                let v: Vec<f64> = (0..p)
                    .map(|col| {
                        if col_range.contains(&col) {
                            sqrt_w * target[[row, col]]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                if whiten {
                    let metric = self.row_metric.as_ref().ok_or_else(|| {
                        "crosscoder_block_ift_rhs: whitening metric absent".to_string()
                    })?;
                    Ok(metric.apply_metric_row(row, ndarray::aview1(&v)))
                } else {
                    Ok(v)
                }
            };
            self.contracted_softmax_linear_rhs(
                cache,
                &second_jets,
                &border,
                probe_for_row,
                |row, q, t_row, beta_row| {
                    let base = cache.row_offsets[row];
                    for (var_idx, &value) in t_row.iter().enumerate().take(q) {
                        t[base + var_idx] = -0.5 * value;
                    }
                    for (channel, &value) in border.iter().zip(beta_row) {
                        beta[channel.index] += -0.5 * value;
                    }
                    Ok(())
                },
            )?;
            return Ok(SaeArrowVector { t, beta });
        }
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let base = cache.row_offsets[row];
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
                .ok_or_else(|| "crosscoder_block_ift_rhs: empty jet window".to_string())?;
            if whiten {
                self.apply_whiten_to_logdet_row_jets(row, &mut jets)?;
            }
            // The non-softmax rank-space dot: jets are whitened to `Uᵀ·`
            // channels, so the vector is whitened the same way (never
            // `M_n v` here — that fold belongs to the contracted path above).
            let sqrt_w = self
                .row_loss_weights
                .as_deref()
                .map_or(1.0, |w| w[row].sqrt());
            let mut v: Vec<f64> = (0..p)
                .map(|col| {
                    if col_range.contains(&col) {
                        sqrt_w * target[[row, col]]
                    } else {
                        0.0
                    }
                })
                .collect();
            if whiten {
                let metric = self.row_metric.as_ref().ok_or_else(|| {
                    "crosscoder_block_ift_rhs: whitening metric absent".to_string()
                })?;
                Self::whiten_logdet_metric_vec(metric, row, p, &mut v)?;
            }
            for var_idx in 0..jets.vars.len() {
                t[base + var_idx] = -0.5 * sae_dot(jets.first(var_idx), &v);
            }
            for (channel_pos, channel) in border.iter().enumerate() {
                beta[channel.index] += -0.5 * sae_dot(jets.beta(channel_pos), &v);
            }
        }
        Ok(SaeArrowVector { t, beta })
    }

    /// Dense reconstruction of the θ-adjoint `Γ_w = tr(inv · ∂H/∂θ_w)` against an
    /// arbitrary dense joint inverse `inv` (`dim×dim` over the `(t, β)` blocks).
    ///
    /// #2234 — `inv` is read on the arrow's positions only, so it may be held as arrow blocks
    /// (the arrow orbit lane's weights); the ordered Beta--Bernoulli leg alone needs it dense.
    pub(crate) fn logdet_theta_adjoint_dense<W: JointWeight + ?Sized>(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        inv: &W,
        skip_deflation_dk: bool,
        exact_a: bool,
        // #2330 Patch D — the data target, required ONLY for the exact-A
        // residual-curvature third-derivative leg `⟨error_metric, ∂³f⟩`. `None`
        // reproduces the pre-Patch-D behaviour exactly (the leg is skipped), so
        // every non-exact-A caller passes `None`.
        residual_target: Option<ArrayView2<'_, f64>>,
    ) -> Result<SaeArrowVector, String> {
        // #2330 — `skip_deflation_dk` drops the Daleckii–Krein deflation
        // correction, leaving the raw trace contraction (a deflation-blind
        // counterfactual for tests). Production callers pass `false`.
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
        let patchd_gate = PatchDGate::for_mode(&self.assignment.mode);
        let patchd_is_obb = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        );
        // Full ordered-BB prior curvature: the row loop below contributes only
        // softmax entropy, so this owner must include both the positive local
        // diagonal and the exact-minus-majorizer remainder.
        let patchd_obb_adjoint = if patchd_residual.is_some() && patchd_is_obb {
            crate::assignment::ordered_beta_bernoulli_logit_adjoint_data_weighted(
                &self.assignment,
                rho,
                self.row_loss_weights.as_deref(),
            )?
        } else {
            None
        };
        // #2933 F03 — `1/τ` of a per-logit sigmoid gate (ordered Beta--Bernoulli,
        // ThresholdGate), whose logit moves only its own atom's reconstruction leg;
        // `0` for the simplex and support gates.
        let independent_gate_inv_tau = crate::assignment::sigmoid_gate_frame(&self.assignment.mode)
            .map_or(0.0, |(_, temperature)| 1.0 / temperature);
        // #2933 F03 — the ThresholdGate prior's logit curvature scales with
        // `λ_sparse`; every other mode's diagonal-prior leg ignores it.
        let threshold_strength = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => rho.lambda_sparse()?,
            _ => 0.0,
        };
        // #2933 F24 — rows holding an embedded-sphere block factor the Riemannian
        // conversion of their ambient row, which moves with θ too (`SphereRowConversion`).
        let sphere_blocks = self.sphere_tangent_blocks_by_row(&cache.row_dims)?;
        let sphere_axis_periods = self.all_ard_axis_periods();
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
            let simplex_count = crate::assignment::simplex_gate_free_count(&self.assignment);
            let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            // #2330 Patch D — per-row `error_metric = √w·M·r` in output space,
            // built EXACTLY as `apply_exact_hessian_minus_b` builds the object it
            // contracts ΔC against (√w residual, then whitening metric applied).
            let patchd_error_metric: Option<Vec<f64>> = patchd_residual.map(|tgt| {
                self.patchd_row_error_metric(row, w_row, tgt, &assignments, whiten_row_jets)
            });
            let patchd_sqrt_w = w_row.sqrt();
            let patchd_reconstruction: Option<PatchDRowContractions> = patchd_error_metric
                .as_deref()
                .zip(patchd_third_jets.as_deref())
                .map(|(em, third_jets)| {
                    self.patchd_row_contractions(row, em, &assignments, &second_jets, third_jets)
                });
            let patchd_ctx: Option<PatchDResidualCtx<'_>> = patchd_error_metric
                .as_deref()
                .zip(patchd_reconstruction.as_ref())
                .map(|(em, reconstruction)| PatchDResidualCtx {
                    row,
                    error_metric: em,
                    sqrt_w: patchd_sqrt_w,
                    assignments: &assignments,
                    second_jets: &second_jets,
                    gate: patchd_gate,
                    reconstruction,
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
            let defl_live = Self::row_deflation_is_live(defl_dirs, defl_spectrum);
            let inv_vv_block = if !defl_live {
                Array2::<f64>::zeros((0, 0))
            } else {
                inv.row_block(base, q)
            };
            // #2933 F24 — on a sphere row the tower keeps its ambient derivatives as
            // matrices, converts them (`SphereRowConversion`), contracts the converted
            // block, and projects the slot functional to the tangent after the loop.
            let sphere_conversion = if sphere_blocks[row].is_empty() {
                None
            } else {
                let target = residual_target.ok_or_else(|| {
                    format!(
                        "logdet_theta_adjoint_dense: row {row} holds an embedded-sphere block, \
                         whose Riemannian conversion reads the row residual, but no target \
                         was supplied"
                    )
                })?;
                let error_metric =
                    self.patchd_row_error_metric(row, w_row, target, &assignments, whiten_row_jets);
                SphereRowConversion::for_row(
                    self,
                    row,
                    &sphere_blocks[row],
                    &jets,
                    border.len(),
                    &error_metric,
                    &ard_precisions,
                    &sphere_axis_periods,
                    exact_a,
                )
            };
            let collect_matrices = defl_live || sphere_conversion.is_some();
            let contract_converted = |d_tt: &Array2<f64>, d_tbeta: &Array2<f64>| -> f64 {
                let mut converted = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        converted += inv.entry(base + b, base + a) * d_tt[[a, b]];
                    }
                }
                if defl_live && !skip_deflation_dk {
                    converted -= Self::deflation_block_correction(
                        &inv_vv_block,
                        d_tt,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, ch) in border.iter().enumerate() {
                        converted +=
                            2.0 * inv.entry(base + a, total_t + ch.index) * d_tbeta[[a, beta_pos]];
                    }
                }
                converted
            };
            let mut sphere_slot_functional =
                Array1::<f64>::zeros(if sphere_conversion.is_some() { q } else { 0 });
            for w in 0..q {
                let logit_w = match jets.vars[w] {
                    SaeLocalRowVar::Logit { atom } => Some(atom),
                    SaeLocalRowVar::Coord { .. } => None,
                };
                // The exact operator's entropy block moves with the logit through the
                // dense entropy third derivative; one O(K) setup per logit variable.
                let softmax_entropy_derivative = match logit_w {
                    Some(atom_w) if exact_a && entropy_scale != 0.0 => Some(
                        SoftmaxEntropyDerivative::new(
                            a_soft,
                            atom_w,
                            m_log_mean,
                            entropy_scale,
                            inv_tau,
                        ),
                    ),
                    _ => None,
                };
                let mut gamma = 0.0_f64;
                let mut dh_mat = if !collect_matrices {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                let mut dh_border = if sphere_conversion.is_some() {
                    Array2::<f64>::zeros((q, border.len()))
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                let mut beta_beta = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = 0.0_f64;
                        dh += match (logit_w, jets.vars[a], jets.vars[b]) {
                            (
                                Some(atom_w),
                                SaeLocalRowVar::Coord { atom: atom_a, .. },
                                SaeLocalRowVar::Coord { atom: atom_b, .. },
                            ) => {
                                // #2330 / #2371 / #2933 F03 -- independent sigmoid gate
                                // gradient of the GN curvature (ordered Beta--Bernoulli
                                // and ThresholdGate). `B[a,b] = <J_a, J_b>` and each leg
                                // `J_k` carries its INDEPENDENT gate
                                // `g_k = sigma((l_k - threshold)/tau)` linearly, so
                                // `dB/dl_w = [1(w==a) + 1(w==b)] * (1-g_w)/tau * B`.
                                // The matching leg gate is `g_w`, so a single
                                // `(1 - a_soft[atom_w])` is correct per side:
                                // same-atom-both gives sided=2, one-sided cross-atom
                                // gives sided=1 (the #2371 term wrongly dropped as
                                // exactly zero). The softmax factor is 0 for every
                                // non-softmax mode and `independent_gate_inv_tau` is 0
                                // for softmax, so each family reads only its own leg.
                                let sided =
                                    (atom_w == atom_a) as u32 + (atom_w == atom_b) as u32;
                                sae_dot(jets.first(a), jets.first(b))
                                    * (Self::softmax_data_weight_product_logit_factor(
                                        a_soft, atom_a, atom_b, atom_w, inv_tau,
                                    ) + sided as f64
                                        * (1.0 - a_soft[atom_w])
                                        * independent_gate_inv_tau)
                            }
                            _ => {
                                sae_dot(jets.second(a, w), jets.first(b))
                                    + sae_dot(jets.first(a), jets.second(b, w))
                            }
                        };
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg(
                                ctx,
                                jets.vars[a],
                                jets.vars[b],
                                jets.vars[w],
                            );
                        }
                        if exact_a {
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
                            // third-jet leg `⟨error_metric, ∂³f_abw⟩` is
                            // `patchd_residual_third_leg` above.
                            dh += sae_dot(jets.first(w), jets.second(a, b));
                        }
                        if let (
                            Some(atom_w),
                            SaeLocalRowVar::Logit { atom: atom_a },
                            SaeLocalRowVar::Logit { atom: atom_b },
                        ) = (logit_w, jets.vars[a], jets.vars[b])
                        {
                            if exact_a {
                                // #2333 — `A` carries the exact dense entropy Hessian on
                                // the logit block: `B`'s Gershgorin majorizer `D̃` plus the
                                // ΔC remainder `h_entropy − D̃` on the diagonal and
                                // `h_entropy` off it. Its logit derivative is the dense
                                // entropy third derivative, off-diagonal pairs included.
                                if let Some(derivative) = softmax_entropy_derivative.as_ref() {
                                    dh += w_row * derivative.entry(atom_a, atom_b).1;
                                }
                            } else if atom_a == atom_b {
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
                            // #2080 — the softmax row's logit Jacobian has the exact dense
                            // curvature `c·(diag z − zzᵀ)/τ²` in both `B` and `A`.
                            if let Some(count) = simplex_count {
                                if !self.assignment.logits_are_fixed() {
                                    dh += w_row
                                        * crate::assignment::simplex_gate_logit_jacobian_third(
                                            a_soft, atom_a, atom_b, atom_w, count, inv_tau,
                                        );
                                }
                            }
                        }
                        if a == b && a == w {
                            // #2080 — the gate prior's logit Jacobian has the exact curvature
                            // `2z(1 − z)/τ²` in both `B` and `A`. #2933 F03 — the
                            // ThresholdGate prior's own logit curvature is the PSD clamp in
                            // `B` and the signed value in `A`, so each operator reads its own
                            // derivative off the shared seam. Softmax's entropy block is
                            // differentiated above and ordered Beta--Bernoulli's prior by
                            // `patchd_obb_adjoint` below, so the seam's `None` channel is
                            // silent for them.
                            if let SaeLocalRowVar::Logit { atom } = jets.vars[a] {
                                dh += self.assignment_prior_hdiag_derivative_entry(
                                    threshold_strength,
                                    row,
                                    atom,
                                    jets.vars[w],
                                    None,
                                    exact_a,
                                );
                            }
                            if let SaeLocalRowVar::Coord { atom, axis } = jets.vars[a] {
                                if !ard_precisions[atom].is_empty() {
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
                        if collect_matrices {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv.entry(base + b, base + a) * dh;
                    }
                }
                if defl_live && !skip_deflation_dk {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv_block,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
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
                        if sphere_conversion.is_some() {
                            dh_border[[a, beta_pos]] = dh;
                        }
                        gamma += 2.0 * inv.entry(base + a, total_t + ch.index) * dh;
                    }
                }
                for (beta_i, ch_i) in border.iter().enumerate() {
                    for (beta_j, ch_j) in border.iter().enumerate() {
                        let dh = sae_dot(jets.beta_deriv(w, beta_i), jets.beta(beta_j))
                            + sae_dot(jets.beta(beta_i), jets.beta_deriv(w, beta_j));
                        let contribution = inv.entry(total_t + ch_i.index, total_t + ch_j.index) * dh;
                        gamma += contribution;
                        beta_beta += contribution;
                    }
                }
                match sphere_conversion.as_ref() {
                    None => gamma_t[base + w] = gamma,
                    Some(conversion) => {
                        let (d_tt, d_tbeta) = conversion.slot_derivative(w, &dh_mat, &dh_border);
                        sphere_slot_functional[w] = contract_converted(&d_tt, &d_tbeta) + beta_beta;
                    }
                }
            }
            if let Some(conversion) = sphere_conversion.as_ref() {
                let projected = conversion.project_slot_functional(&sphere_slot_functional);
                for w in 0..q {
                    gamma_t[base + w] = projected[w];
                }
            }
            for (w_beta_pos, w_channel) in border.iter().enumerate() {
                let mut gamma = 0.0_f64;
                let mut dh_mat = if !collect_matrices {
                    Array2::<f64>::zeros((0, 0))
                } else {
                    Array2::<f64>::zeros((q, q))
                };
                let mut dh_border = if sphere_conversion.is_some() {
                    Array2::<f64>::zeros((q, border.len()))
                } else {
                    Array2::<f64>::zeros((0, 0))
                };
                for a in 0..q {
                    for b in 0..q {
                        let mut dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.first(b))
                            + sae_dot(jets.first(a), jets.beta_l_deriv(b, w_beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.beta(w_beta_pos), jets.second(a, b));
                        }
                        if let Some(ctx) = patchd_ctx.as_ref() {
                            dh += self.patchd_residual_third_leg_beta(
                                ctx, jets.vars[a], jets.vars[b], w_channel,
                            );
                        }
                        if collect_matrices {
                            dh_mat[[a, b]] = dh;
                        }
                        gamma += inv.entry(base + b, base + a) * dh;
                    }
                }
                if defl_live && !skip_deflation_dk {
                    gamma -= Self::deflation_block_correction(
                        &inv_vv_block,
                        &dh_mat,
                        defl_dirs,
                        defl_spectrum,
                    );
                }
                for a in 0..q {
                    for (beta_pos, ch) in border.iter().enumerate() {
                        let mut dh = sae_dot(jets.beta_l_deriv(a, w_beta_pos), jets.beta(beta_pos));
                        if exact_a {
                            dh += sae_dot(jets.beta(w_beta_pos), jets.beta_deriv(a, beta_pos));
                        }
                        if sphere_conversion.is_some() {
                            dh_border[[a, beta_pos]] = dh;
                        }
                        gamma += 2.0 * inv.entry(base + a, total_t + ch.index) * dh;
                    }
                }
                match sphere_conversion.as_ref() {
                    None => gamma_beta[w_channel.index] += gamma,
                    Some(conversion) => {
                        let (d_tt, d_tbeta) =
                            conversion.border_derivative(w_beta_pos, &dh_mat, &dh_border);
                        gamma_beta[w_channel.index] += contract_converted(&d_tt, &d_tbeta);
                    }
                }
            }
        }
        if exact_a {
            let border = inv.border_block(total_t).ok_or_else(|| {
                format!("logdet_theta_adjoint_dense: the weight holds no border after {total_t} coordinates")
            })?;
            gamma_beta += &self.exact_decoder_prior_theta_trace(cache, border)?;
        }
        // Fold the entire ordered-BB prior derivative into the logit slots.
        if let Some(data) = patchd_obb_adjoint.as_ref() {
            let inv = inv.dense().ok_or_else(|| {
                "logdet_theta_adjoint_dense: the ordered Beta--Bernoulli prior leg reads cross-row \
                 entries, which an arrow-held weight does not carry"
                    .to_string()
            })?;
            let obb = self.dense_exact_a_ordered_bb_logit_theta_adjoint(cache, inv, data)?;
            gamma_t += &obb;
        }
        Ok(SaeArrowVector {
            t: gamma_t,
            beta: gamma_beta,
        })
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
    ///
    /// #2267 — the dense route differentiates the value its evaluation priced off
    /// `dense_geometry`, the spectral block that value was classified on, and reads every
    /// eigensystem it needs from it. The streaming route carries no dense block.
    pub(crate) fn analytic_outer_rho_gradient_components_with_bundle(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        evidence: Option<BundleEvidenceGeometry<'_>>,
        matrix_free_system: Option<&ArrowSchurSystem>,
        dense_geometry: Option<&DenseExactAGeometry>,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        self.analytic_outer_rho_gradient_components_on_route(
            target,
            rho,
            loss,
            cache,
            solver,
            evidence,
            matrix_free_system,
            dense_geometry.map(ExactAGeometry::Dense),
        )
    }

    /// #2234 step 1a — [`Self::analytic_outer_rho_gradient_components_with_bundle`] on the arrow
    /// orbit lane: the value was priced off `geometry`, and its log-determinant channels and
    /// stationarity adjoint are read off it, as the dense route's are read off its block.
    pub(crate) fn analytic_outer_rho_gradient_components_arrow_orbit(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        geometry: &ArrowOrbitGeometry,
    ) -> Result<SaeOuterRhoGradientComponents, OuterGradientError> {
        self.analytic_outer_rho_gradient_components_on_route(
            target,
            rho,
            loss,
            cache,
            solver,
            None,
            None,
            Some(ExactAGeometry::ArrowOrbit(geometry)),
        )
    }

    fn analytic_outer_rho_gradient_components_on_route(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        loss: &SaeManifoldLoss,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        evidence: Option<BundleEvidenceGeometry<'_>>,
        matrix_free_system: Option<&ArrowSchurSystem>,
        exact_geometry: Option<ExactAGeometry<'_>>,
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
        // No bundle ⇒ the dense exact-A pseudo-inverse built on the caller's
        // `cache` (`dense_exact_a_logdet_channels`). A bundle ⇒ the exact observed
        // information, carried with its own cache.
        let logdet_derivative_bundle = evidence
            .as_ref()
            .map(|geometry| (geometry.probes, geometry.sinv));
        let evidence_cache = evidence.as_ref().map_or(cache, |geometry| geometry.cache);
        let evidence_operator = evidence
            .as_ref()
            .map_or(EvidenceOperator::Majorizer, |geometry| geometry.operator);
        // #2933 F03 — the derivative accepts only the identity of the value it
        // differentiates. Both production value routes rank the exact observed
        // information `½log|A|`: the dense direct-logdet criterion off one joint
        // eigensystem, which its evaluation hands here as `dense_geometry` (no bundle,
        // no system), and the streaming criterion off `exact_a_evidence_system`, whose
        // artifact hands this assembler a bundle naming that operator TOGETHER with its
        // matrix-free system. Every other pairing — a system without its bundle, a
        // bundle without its system, or a bundle naming the majorizer — would contract
        // `B` channels, or the inverses of two different operators, against an
        // `A`-valued score. A dense route without its block (#2267) would decompose `A`
        // again for each consumer. Refuse them.
        match (evidence.as_ref(), matrix_free_system, exact_geometry) {
            (None, None, Some(_)) => {}
            (Some(geometry), Some(_), None) if geometry.operator.is_exact_a() => {}
            (bundle, system, dense) => {
                return Err(OuterGradientError::internal(format!(
                    "analytic_outer_rho_gradient_components_with_bundle: the criterion value \
                     ranks the exact observed information ½log|A|, but this derivative route \
                     pairs evidence operator {:?} with matrix-free system present = {} and \
                     dense spectral block present = {}. Only the dense exact-A route (the \
                     evaluation's spectral block, no bundle, no system) and the streaming \
                     exact-A route (the bundle naming ExactObservedInformation with its \
                     system, no dense block) differentiate that value.",
                    bundle.map(|geometry| geometry.operator),
                    system.is_some(),
                    dense.is_some(),
                )));
            }
        }
        let n_params = rho.flat_coordinates().len();
        let mut explicit = Array1::<f64>::zeros(n_params);
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        let mut occam = Array1::<f64>::zeros(n_params);
        let mut third_order_correction = Array1::<f64>::zeros(n_params);
        let rank_charge = self
            .production_rank_charge_derivative(
                target,
                rho,
                loss,
                cache,
                exact_geometry.and_then(ExactAGeometry::dense),
            )
            .map_err(OuterGradientError::internal)?;
        // #2330 Phase-2 / #2333 — which operator the logdet channels belong to
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
        //
        // #2933 F03 — this used to also require a family predicate
        // (`dense_exact_a_theta_adjoint_is_modelled`) that excluded ThresholdGate,
        // because `logdet_theta_adjoint_dense` lacked the independent-sigmoid
        // data-weight leg and the gate prior's signed logit curvature. Those legs
        // now exist, and the dense exact-A reconstruction models every family, so
        // the route alone decides. Before, a dense ThresholdGate fit ranked ½log|A|
        // and returned `½tr(B⁻¹∂B)` channels from cached `B` geometry.
        let exact_a_logdet_route = logdet_derivative_bundle.is_none();

        // #2087/#2330 ROUTE-COHERENCE GUARD. The VALUE's log-determinant route and
        // THIS gradient's are selected by two unrelated predicates:
        //
        //   value    `streaming_plan().admitted_or_error(..).direct_logdet_admitted()`
        //            — a working-set/memory admission (construction_quasi_laplace.rs).
        //   gradient `exact_a_logdet_route` above — the bundle / matrix-free pair.
        //            Nothing consults the value's admission.
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
        // to a value-side identity check, which re-derives the VALUE predicate and
        // so cannot observe that the gradient took the other route. Refuse rather
        // than return the derivative of an operator the value never ranked.
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
        //
        // #2234 — the arrow orbit lane is bundle-free too, and its value is the streaming one
        // by construction, so the guard reads the plan only for the dense block.
        let dense_block_route = !matches!(exact_geometry, Some(ExactAGeometry::ArrowOrbit(_)));
        if exact_a_logdet_route && dense_block_route {
            let value_route_is_exact_a = self
                .streaming_plan()
                .map_err(OuterGradientError::internal)?
                .admitted_or_error(self.n_obs(), self.output_dim(), self.k_atoms())
                .map_err(OuterGradientError::internal)?
                .direct_logdet_admitted();
            if !value_route_is_exact_a {
                return Err(OuterGradientError::internal(format!(
                    "analytic_outer_rho_gradient_components_with_bundle: log-determinant route \
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
                logdet_trace[sparse_index] = joint_trace;
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
                            "analytic_outer_rho_gradient_components_with_bundle: smooth dof (matrix-free): {err}"
                        ),
                    })?,
                None => self
                    .decoder_smoothness_effective_dof_with_solver_per_atom(
                        cache,
                        solver,
                        &lambda_smooth_vec,
                    )
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!("analytic_outer_rho_gradient_components_with_bundle: {err}"),
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
        // #2933 F26 — a curvature-parameterised penalty moves the prior normalizer
        // `½·r·log|S(κ)|_+`, so its κ coordinate carries an Occam channel too.
        for (index, derivative) in self
            .reml_occam_kappa_derivative(rho)
            .map_err(OuterGradientError::internal)?
        {
            occam[index] = -derivative;
        }
        // #2935 — the same curvature moves the penalty energy `½λ<B, S(κ) B>`, so the κ
        // coordinate carries `½λ<B, ∂S/∂κ B>` on the scale `loss.smoothness` is priced
        // at (the smoothing entries' scale; an energy that is zero on every atom leaves
        // that scale unobserved and the entries above unscaled). Off the dense exact-A
        // route it also carries `½tr(A⁻¹ ∂A/∂κ)` from the one probe bundle; on that
        // route `dense_exact_a_logdet_channels` supplies it, because its operator map
        // already carries `λ·∂S/∂κ ⊗ I`.
        let smooth_energy_scale = if smooth_explicit_sum.abs() > 0.0 {
            loss.smoothness / smooth_explicit_sum
        } else {
            1.0
        };
        for (index, derivative) in self
            .decoder_smoothness_kappa_energy_derivatives(rho, &lambda_smooth_vec)
            .map_err(OuterGradientError::internal)?
        {
            explicit[index] += smooth_energy_scale * derivative;
        }
        if let Some((probes, sinv)) = logdet_derivative_bundle {
            for (index, trace) in self
                .decoder_kappa_penalty_trace_from_probes(probes, sinv, rho, &lambda_smooth_vec)
                .map_err(|err| OuterGradientError::InternalInvariant {
                    reason: format!(
                        "analytic_outer_rho_gradient_components_with_bundle: curvature logdet \
                         trace (matrix-free): {err}"
                    ),
                })?
            {
                logdet_trace[index] = 0.5 * trace;
            }
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
                            "analytic_outer_rho_gradient_components_with_bundle: ARD logdet trace \
                             (matrix-free): {err}"
                        ),
                    })?,
                None => self
                    .ard_log_precision_hessian_trace(rho, cache, solver, evidence_operator)
                    .map_err(|err| OuterGradientError::InternalInvariant {
                        reason: format!("analytic_outer_rho_gradient_components_with_bundle: {err}"),
                    })?,
            };
            Some(joint)
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
                if let Some(joint) = ard_logdet_traces.as_ref() {
                    logdet_trace[idx] += joint[k][axis];
                }
            }
        }

        // The scalar criterion adds the realised-rank charge to `½ log|H|`.
        // Its direct rho differential belongs alongside the explicit
        // penalty channels and is present on every layout (dense or probes).
        explicit += &rank_charge.direct_rho;

        // #2080: the envelope Γ off the SAME shared low-rank logdet derivative
        // representation (the all-or-nothing cluster's third channel). #2712: the
        // border-only bundle reconstructs the row block on the DEFLATED chart too —
        // `A_i` is the conditioned row Cholesky, so `A_i⁻¹ + G_i S⁻¹ G_iᵀ` is the
        // deflated `(H⁻¹)_tt` — and `logdet_theta_adjoint_from_probes` subtracts the
        // same Daleckii–Krein correction the dense route subtracts instead of routing
        // the fit away. Ordered Beta--Bernoulli uses its row-local PSD majorizer
        // and shared-mass derivative directly.
        // This completes the matrix-free selected-inverse cluster (smoothness EDF + ARD
        // Hessian trace + θ-adjoint); assignment log-strength traces remain
        // solver-bound
        // — the last gaps before the routing flip (see the docstring).
        //
        // #2333 — a bundle is the only producer here. The pairing refusal above
        // admits no bundle-free route other than the dense exact-A one, whose Γ
        // `dense_exact_a_logdet_channels` builds below, so the Trace-seam majorizer
        // adjoint that used to be the bundle-free arm had no route left to serve.
        let majorizer_gamma = match logdet_derivative_bundle {
            None => None,
            Some((probes, sinv)) => Some(
                self.logdet_theta_adjoint_from_probes(
                    rho,
                    evidence_cache,
                    probes,
                    sinv,
                    evidence_operator,
                    Some(target),
                )
                .map_err(OuterGradientError::internal)?,
            ),
        };
        // `½ Γ_joint·theta_hat + ∇R·theta_hat` is represented by one effective
        // logdet adjoint `Γ_eff = Γ_joint + 2∇R`, preserving the existing
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
        // `solve_exact_stationarity` applies the spectral pseudoinverse of
        // `A = B + ΔC`, where
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
        // deflation uses Euclidean spectral projections, so `A⁺` is
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
        // — but no longer which operator is priced.
        //
        // Exactly one arm produces Γ, so the two assemblies cannot both be paid
        // for on one gradient.
        let (gamma, dense_stationarity_adjoint) = match majorizer_gamma {
            Some(gamma) => (gamma, None),
            None => {
                let geometry = exact_geometry.ok_or_else(|| {
                    OuterGradientError::internal(
                        "analytic_outer_rho_gradient_components_with_bundle: the dense exact-A \
                         log-determinant channels need the evaluation's spectral block"
                            .to_string(),
                    )
                })?;
                let DenseExactALogdetChannels {
                    logdet_trace: exact_logdet_trace,
                    theta_adjoint: exact_gamma,
                    stationarity_adjoint,
                } = match geometry {
                    ExactAGeometry::Dense(geometry) => self.dense_exact_a_logdet_channels(
                        target,
                        rho,
                        cache,
                        geometry,
                        &rank_charge.theta,
                    ),
                    ExactAGeometry::ArrowOrbit(geometry) => self.arrow_orbit_logdet_channels(
                        target,
                        rho,
                        cache,
                        geometry,
                        &rank_charge.theta,
                    ),
                }
                .map_err(OuterGradientError::internal)?;
                logdet_trace = exact_logdet_trace;
                (exact_gamma, Some(stationarity_adjoint))
            }
        };

        // At massive K (`matrix_free_system = Some`) the materialized operator is
        // unavailable, so the adjoint uses the certified Ritz pseudoinverse
        // route. Every dense arm is owned by the rank-revealing spectral
        // pseudoinverse. On the exact-A logdet arm it reuses the eigensystem that
        // produced Γ; on a B-majorizer arm it materializes that same physical A
        // once here. Both representations use the same spectral null policy.
        let adjoint = match (matrix_free_system, dense_stationarity_adjoint) {
            (Some(system), None) => {
                self.solve_exact_stationarity_matrix_free(rho, target, cache, system, &gamma)
            }
            (None, Some(adjoint)) => Ok(adjoint),
            (None, None) => self.solve_exact_stationarity(rho, target, cache, &gamma),
            (Some(_), Some(_)) => Err(
                "analytic_outer_rho_gradient_components_with_bundle: dense exact-A adjoint was assembled \
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
                            "analytic_outer_rho_gradient_components_with_bundle: rho carries block \
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
            logdet_trace,
            occam,
            third_order_correction,
        })
    }

    /// Classification of the exact observed information `A = B + ΔC` for the
    /// value path (#2330 / #2336 / #2673 / #2933 F07).
    ///
    /// A converged inner mode is a genuine exact-Laplace maximum iff every
    /// direction of the pencil `(A, Φ)` is `≥ −floor` in its own band, and that band
    /// is [`ExactHessianSpectralBlock::rank_floor`] — ONE predicate, in the ONE
    /// metric the gradient path classifies the same directions by. #2330 ACCEPTS
    /// above `−floor`; #2336's clamp attribution TRIGGERS below it; both read
    /// that one function, so they cannot disagree in the band.
    ///
    /// **Two rules this used to be are gone.** #2673 retired `1e-9 · max(λ_max(A), 1)`,
    /// a single number applied to every direction, whose band moved when the units of
    /// `θ = (t, β)` did (`tests::the_two_floors_are_incommensurable_thresholds_on_one_operator_2673`
    /// records what that pair did). #2933 F07 retired its successor, `|λᵢ| ≤ √ε·vᵢᵀBvᵢ`
    /// on ordinary eigenvectors of `A`, which a nonorthogonal change of coordinates moved
    /// too. See [`sae_exact_a_pencil_floor`] for the argument.
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
    /// Exact-A evidence value, with one coherent basin matrix on each resolved
    /// negative spectral subspace. The same owner supplies the differential
    /// consumed by the outer gradient.
    ///
    /// The spectral block the value was priced on is returned beside it, with the pricing
    /// inputs it was classified with, so the dense evaluation hands it to the dispersion's
    /// fitted-response divergence and to every derivative consumer instead of materializing
    /// and decomposing `A` again (#2933 F36, #2267).
    pub(crate) fn exact_observed_information_log_dets_with_saddle_directions(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        saddle_directions: &mut Vec<(Array1<f64>, f64)>,
    ) -> Result<(f64, DenseExactAGeometry), SaeCriterionError> {
        let geometry = self.materialize_dense_exact_a_geometry(rho, target, cache)?;
        let joint = &geometry.block;
        // #2080/#2267 — a refused basin pushes its refused directions here: `Φ`-normalized
        // vectors in the joint `(t, β)` cache layout, each with its basin curvature, most
        // negative first. The evidence root descends them before it concludes the state
        // has no Laplace normaliser, off this same eigensystem, so a refusal pays one dense
        // materialization and eigendecomposition, not two.
        let joint_pricing = Self::classify_exact_hessian_basin(
            joint,
            &geometry.e_diag,
            geometry.e_beta.as_ref(),
            geometry.total_t,
            "joint",
            Some(saddle_directions),
        )?;
        // #2267 — the priced rank stratum, once per evaluation. Trace job 578389 read
        // sae_manifold_euclidean_k2_terminates price its incumbent at −3.591e3 while trial
        // points 0.125, 0.0625 and 0.03125 away priced +5.468e2, +5.536e2 and +5.570e2,
        // and the inner objective moved continuously (−45.10, −38.46, −35.14 → −31.81).
        // A jump that size over a continuous state is a change in which directions
        // ½log|A| prices: an in-band direction adds its metric curvature, a retained one
        // adds ½·ln μ on top of it. `substituted` counts the in-band directions only the
        // evidence factor's substituted stiffness puts in the band, `resolution_limited`
        // the ones only their numerical resolution does.
        let floor = sae_exact_a_pencil_floor();
        let mut retained = 0usize;
        let mut in_band = 0usize;
        let mut substituted = 0usize;
        let mut resolution_limited = 0usize;
        let mut min_retained_over_floor = f64::INFINITY;
        let mut max_band_over_floor = 0.0_f64;
        for index in 0..joint.eigenvalues.len() {
            let magnitude = joint.eigenvalues[index].abs();
            let edge = joint.rank_floor(index);
            if magnitude <= edge {
                in_band += 1;
                max_band_over_floor = max_band_over_floor.max(magnitude / edge);
                let bare = floor.max(joint.resolution[index]);
                if magnitude > bare {
                    substituted += 1;
                } else if magnitude > floor {
                    resolution_limited += 1;
                }
            } else if joint.eigenvalues[index] > 0.0 {
                retained += 1;
                min_retained_over_floor = min_retained_over_floor.min(magnitude / edge);
            }
        }
        log::info!(
            "[SAE-EXACT-DENSE] priced: dim={} retained={retained} in_band={in_band} \
             substituted={substituted} resolution_limited={resolution_limited} negative={} \
             ½log|A|={:.6e} ½log|Φ|={:.6e} min retained μ/floor={:.3e} \
             max in-band |μ|/floor={:.3e}",
            joint.eigenvalues.len(),
            joint_pricing.negative.len(),
            0.5 * joint_pricing.log_det,
            0.5 * joint.metric_log_det,
            min_retained_over_floor,
            max_band_over_floor,
        );
        // #2234 — each orbit coordinate integrated exactly: `½log|A|` becomes
        // `½log|A_s| − ½log det N − Σ log I_k + ½K·log 2π`, priced only once the complement
        // classified without refusal, so every coupling form is nonnegative.
        let orbit_correction = if geometry.orbit_generators.is_empty() {
            0.0
        } else {
            let values = Self::price_compact_orbits(&geometry.orbit_generators, &geometry.block)?;
            for (generator, value) in geometry.orbit_generators.iter().zip(values.orbits.iter()) {
                log::info!(
                    "[SAE-EXACT-ORBIT] atom={} priced: nodes={} log I={:.6e} log det N={:.6e} \
                     coupling=[{:.3e}, {:.3e}, {:.3e}] correction={:.6e}",
                    generator.atom,
                    value.integral.angles.len(),
                    value.integral.log_integral,
                    values.log_gram_det,
                    value.coupling_forms[0],
                    value.coupling_forms[1],
                    value.coupling_forms[2],
                    values.log_det_correction,
                );
            }
            if values.orbits.len() > 1 {
                // The block separation makes every cross-orbit complement form zero; the largest
                // one, relative to its diagonal forms, is the rounding the product integral drops.
                let mut largest = 0.0_f64;
                for (left, x) in values.orbits.iter().enumerate() {
                    for y in &values.orbits[left + 1..] {
                        let (u, v) = &x.trigonometric;
                        let (complement_u, complement_v) = &y.complement_images;
                        let diagonal = (x.coupling_forms[0].abs() + x.coupling_forms[2].abs())
                            .sqrt()
                            * (y.coupling_forms[0].abs() + y.coupling_forms[2].abs()).sqrt();
                        let cross = u
                            .dot(complement_u)
                            .abs()
                            .max(u.dot(complement_v).abs())
                            .max(v.dot(complement_u).abs())
                            .max(v.dot(complement_v).abs());
                        if diagonal > 0.0 {
                            largest = largest.max(cross / diagonal);
                        }
                    }
                }
                log::info!(
                    "[SAE-EXACT-ORBIT] {} separated orbits: largest cross/diagonal complement form {largest:.3e}",
                    values.orbits.len()
                );
            }
            values.log_det_correction
        };
        Ok((joint_pricing.log_det + orbit_correction, geometry))
    }

    /// The generalized eigensystem of one already-materialized exact-Hessian block in the
    /// metric `Φ`, and the per-direction band every consumer reads (#2933 F07).
    ///
    /// The pencil is reduced by whitening: `Ã = L⁻¹AL⁻ᵀ` for `Φ = LLᵀ`, `Ã = U diag(μ)Uᵀ`, and
    /// `W = L⁻ᵀU`, so `WᵀΦW = I` and `AW = ΦW diag(μ)`. One predicate owns the classification:
    /// [`Self::rank_floor`], which the matrix-free Ritz route applies to its own pencil Ritz
    /// pairs.
    ///
    /// #2674 — this used to take an analytic chart-gauge basis and diagonalize
    /// `Zᵀ A Z` on its orthogonal complement, deleting the declared orbit
    /// STRUCTURALLY so a spectral floor never had to rediscover it. That made
    /// two null predicates own one eigenvalue array, and the declaration-based
    /// one is the one that is wrong here: the chart orbit is a symmetry of the
    /// RECONSTRUCTION (measured reconstruction-invariant to `rel_ls ~1e-16`),
    /// not of the PENALIZED objective this operator is the Hessian of — the ARD
    /// and smoothing priors are not invariant along it. Where the orbit really is flat
    /// for the penalized operator its `μ` lands in `[−floor(i), floor(i)]` and the
    /// pseudoinverse discards it, which is what the independent oracle in
    /// `exact_observed_information_log_det_matches_the_pencil_oracle_at_a_pd_root_2330` reads.
    fn exact_hessian_spectral_block(
        operator: Array2<f64>,
        metric: &dyn ExactAPencilMetric,
    ) -> Result<ExactHessianSpectralBlock, String> {
        let dimension = operator.nrows();
        if operator.ncols() != dimension || metric.dim() != dimension {
            return Err(format!(
                "exact_hessian_spectral_block: operator {:?}, metric dimension {}",
                operator.dim(),
                metric.dim()
            ));
        }
        // #2267 — the other half of the split; see `materialize_exact_hessian_dense`.
        let eigh_started = std::time::Instant::now();
        // `L⁻¹A`; `A` is symmetric, so `L⁻¹` on the columns of its transpose is `L⁻¹AL⁻ᵀ`.
        let half = metric.lower_solve(operator.view())?;
        let mut whitened = metric.lower_solve(half.t())?;
        drop(half);
        for row in 0..dimension {
            for column in (row + 1)..dimension {
                let average = 0.5 * (whitened[[row, column]] + whitened[[column, row]]);
                whitened[[row, column]] = average;
                whitened[[column, row]] = average;
            }
        }
        let (mut eigenvalues, rotation) = whitened
            .eigh(Side::Lower)
            .map_err(|error| format!("exact_hessian_spectral_block: whitened eigh failed: {error:?}"))?;
        drop(whitened);
        let mut eigenvectors = metric.lower_transpose_solve(rotation.view())?;
        drop(rotation);
        EXACT_A_PENCIL_DECOMPOSITIONS.with(|count| count.set(count.get() + 1));
        log::info!(
            "[SAE-EXACT-DENSE] pencil eigendecomposition DONE: dim={dimension}, {:.3} s, \
             decomposition {} on this thread",
            eigh_started.elapsed().as_secs_f64(),
            exact_a_pencil_decompositions_on_this_thread(),
        );
        let curvature_norm = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        // Rank depends on the substituted stiffness, so a repeated pencil eigenspace must
        // resolve its substitution directions too. The matrix-free route uses this same
        // convention.
        canonicalize_exact_a_rank_clusters(
            &mut eigenvalues,
            &mut eigenvectors,
            curvature_norm,
            &|v| metric.substituted_image(v.view()),
        )?;
        let operator_frobenius = operator.iter().map(|value| value * value).sum::<f64>().sqrt();
        let metric_frobenius = metric.frobenius_norm()?;
        let mut substituted_stiffness = Array1::<f64>::zeros(dimension);
        let mut resolution = Array1::<f64>::zeros(dimension);
        for index in 0..dimension {
            let direction = eigenvectors.column(index);
            let substituted = direction.dot(&metric.substituted_image(direction)?);
            if !substituted.is_finite() {
                return Err(format!(
                    "exact_hessian_spectral_block: direction {index} of the {dimension}-dimensional \
                     block has a non-finite substituted stiffness {substituted:e}"
                ));
            }
            // A direction whose pins lower curvature more than they raise it carries no
            // substituted stiffness, so the total is clamped at zero.
            substituted_stiffness[index] = substituted.max(0.0);
            resolution[index] = sae_exact_a_pencil_resolution(
                dimension,
                direction.dot(&direction),
                operator_frobenius,
                metric_frobenius,
                eigenvalues[index],
            );
        }
        let metric_log_det = metric.log_det()?;
        let band: Vec<usize> = (0..dimension)
            .filter(|&index| {
                eigenvalues[index].abs()
                    <= sae_exact_a_band_edge(
                        eigenvalues[index],
                        resolution[index],
                        substituted_stiffness[index],
                    )
            })
            .collect();
        let mut band_metric_images = Array2::<f64>::zeros((dimension, band.len()));
        for (position, &index) in band.iter().enumerate() {
            band_metric_images
                .column_mut(position)
                .assign(&metric.apply(eigenvectors.column(index))?);
        }
        let block = ExactHessianSpectralBlock {
            operator,
            eigenvalues,
            eigenvectors,
            substituted_stiffness,
            resolution,
            metric_log_det,
            operator_frobenius,
            metric_frobenius,
            band,
            band_metric_images,
            orbit: None,
        };
        let crossings = block.resolution_band_crossings();
        if crossings > 0 {
            // Report when the working coordinates, rather than the pencil, determine the
            // shared spectral rank.
            let widest = block.resolution.iter().copied().fold(0.0_f64, f64::max);
            log::warn!(
                "[SAE-EXACT-DENSE] numerical resolution limit: {crossings} of {dimension} \
                 pencil directions clear √ε but not their own numerical resolution (widest \
                 {widest:.6e}); value and adjoint discard these directions",
            );
        }
        Ok(block)
    }

    /// Price the full basin on the pencil's resolved negative subspace (#2933 F07).
    /// If `W_N` spans that subspace with `W_NᵀΦW_N = I`, `C = W_NᵀAW_N + W_NᵀEW_N`. Its
    /// determinant and definiteness are invariant under `W_N → W_N R`, including repeated
    /// negative eigenvalues. Independent diagonal prices discard E's coupling and do
    /// not even define a continuous value at a repeated eigenvalue.
    ///
    /// The value is
    ///
    /// ```text
    ///   log|A|_reg = log|Φ| + Σ_{μᵢ > floorᵢ} ln μᵢ + Σ_{κⱼ > floorⱼ} ln κⱼ
    /// ```
    ///
    /// with `κ` the basin curvatures: the log-determinant of `ΦW diag(μ̃) WᵀΦ`, in which
    /// every in-band direction, of the pencil or of the basin, carries the metric's own
    /// curvature `μ̃ = 1`. Those directions are integrated against the majorizer's local
    /// Gaussian rather than dropped or given an improper constant. Under `θ → Lθ` the
    /// value moves by `2 log|det L|`, as the log-determinant of any Hessian does, and on a
    /// full-rank positive stratum it IS `log|A|`.
    ///
    /// On a fixed rank stratum `Q = V C⁺ Vᵀ` is the E derivative; the differential adds
    /// the retained inverse, the band's `dΦ` and the response of the negative subspace.
    ///
    /// A basin direction with `κ < −floor` refuses the block. With
    /// `refused_directions` present, every such direction is pushed with its
    /// `κ` (most negative first) before the refusal is returned, so a caller can
    /// descend the saddle rather than only learn that one exists (#2080). The pushed
    /// vectors are `Φ`-normalized, and `κ` is the basin curvature along the vector as
    /// pushed.
    fn classify_exact_hessian_basin(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        label: &'static str,
        mut refused_directions: Option<&mut Vec<(Array1<f64>, f64)>>,
    ) -> Result<ExactHessianBasin, SaeCriterionError> {
        let dim = block.eigenvalues.len();
        if block.eigenvectors.dim() != (dim, dim) || total_t > dim || e_diag.len() < total_t {
            return Err(SaeCriterionError::Numerical(
                "exact-A pricing dimensions disagree".to_string(),
            ));
        }
        let negative: Vec<usize> = (0..dim)
            .filter(|&i| block.eigenvalues[i] < -block.rank_floor(i))
            .collect();
        let complement: Vec<usize> = (0..dim)
            .filter(|&i| block.eigenvalues[i] >= -block.rank_floor(i))
            .collect();
        let mut log_det = block.metric_log_det;
        for &i in &complement {
            let mu = block.eigenvalues[i];
            if mu <= block.rank_floor(i) {
                continue;
            }
            log_det += mu.ln();
        }
        if negative.is_empty() {
            return Ok(ExactHessianBasin {
                log_det,
                negative,
                complement,
                basis: Array2::zeros((dim, 0)),
                rotation: Array2::zeros((0, 0)),
                vectors: Array2::zeros((dim, 0)),
                inverse_values: Array1::zeros(0),
            });
        }
        let q = negative.len();
        let basis = Array2::from_shape_fn((dim, q), |(row, col)| {
            block.eigenvectors[[row, negative[col]]]
        });
        let mut basin = Array2::<f64>::zeros((q, q));
        for row in 0..total_t {
            for i in 0..q {
                let left = e_diag[row] * basis[[row, i]];
                for j in 0..q {
                    basin[[i, j]] += left * basis[[row, j]];
                }
            }
        }
        if e_beta.is_some() {
            // #2828 — the β-tier decoder priors' majorization gap. Dense on the
            // border block, so unlike the coordinate clamps it cannot be folded
            // in as a diagonal weight.
            let border = Self::dropped_curvature_border_forms(
                e_beta,
                total_t,
                basis.view(),
                basis.view(),
            );
            for i in 0..q {
                for j in 0..q {
                    basin[[i, j]] += border[[i, j]];
                }
            }
        }
        for (i, &index) in negative.iter().enumerate() {
            basin[[i, i]] += block.eigenvalues[index];
        }
        let (basin_values, basin_rotation) = basin
            .eigh(Side::Lower)
            .map_err(|error| format!("exact-A basin eigendecomposition: {error:?}"))?;
        let basin_vectors = basis.dot(&basin_rotation);
        // `E` enters `C` at its own scale, so its norm joins the operator's in the
        // resolution of each basin curvature.
        let e_frobenius = (e_diag.iter().take(total_t).map(|x| x * x).sum::<f64>()
            + e_beta.map_or(0.0, |gap| gap.iter().map(|x| x * x).sum::<f64>()))
        .sqrt();
        let mut inverse_values = Array1::<f64>::zeros(q);
        let mut refused = false;
        for (i, &kappa) in basin_values.iter().enumerate() {
            let vector = basin_vectors.column(i);
            if !kappa.is_finite() {
                return Err(SaeCriterionError::Numerical(
                    "exact-A basin needs a finite curvature".to_string(),
                ));
            }
            let resolution = sae_exact_a_pencil_resolution(
                dim,
                vector.dot(&vector),
                block.operator_frobenius + e_frobenius,
                block.metric_frobenius,
                kappa,
            );
            let floor = sae_exact_a_band_edge(kappa, resolution, 0.0);
            if kappa < -floor {
                log::warn!(
                    "SAE exact-A basin refusal: block={label}, mode={i}, curvature={kappa:e}, floor={floor:e}"
                );
                let Some(directions) = refused_directions.as_mut() else {
                    return Err(SaeCriterionError::IndefiniteObservedInformation { block: label });
                };
                directions.push((vector.to_owned(), kappa));
                refused = true;
                continue;
            }
            if kappa <= floor {
                continue;
            }
            log_det += kappa.ln();
            inverse_values[i] = 1.0 / kappa;
        }
        if refused {
            return Err(SaeCriterionError::IndefiniteObservedInformation { block: label });
        }
        Ok(ExactHessianBasin {
            log_det,
            negative,
            complement,
            basis,
            rotation: basin_rotation,
            vectors: basin_vectors,
            inverse_values,
        })
    }

    /// Realize the differential only for callers that consume it. Scalar
    /// evidence evaluation stops at the classified basin spectrum above.
    fn price_exact_hessian_block(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        label: &'static str,
    ) -> Result<ExactHessianPricing, SaeCriterionError> {
        let basin =
            Self::classify_exact_hessian_basin(block, e_diag, e_beta, total_t, label, None)?;
        Self::exact_hessian_basin_differential(block, e_diag, e_beta, total_t, &basin)
    }

    /// The differential of [`Self::classify_exact_hessian_basin`]'s value on its rank
    /// stratum (#2933 F07), as three contraction weights:
    ///
    /// ```text
    ///   d log|A|_reg = ⟨X_A, dA⟩ + ⟨X_Φ, dΦ⟩ + ⟨X_E, dE⟩
    ///   X_A = W_P M_P⁻¹ W_Pᵀ + U_p K_p⁻¹ U_pᵀ + sym(2 W_N Ĥ W_cᵀ)
    ///   X_Φ = W_Z W_Zᵀ + U_z U_zᵀ − sym(2 W_N M_N Ĥ W_cᵀ)
    ///   X_E = U_p K_p⁻¹ U_pᵀ
    /// ```
    ///
    /// `P`/`Z`/`N` are the retained, in-band and negative pencil directions, `c = P ∪ Z` the
    /// complement, `U = W_N R` the basin vectors split into priced (`p`, curvatures `K_p`)
    /// and in-band (`z`), and `Ĥ = (C⁺ W_NᵀEW_c) ⊘ (μ_N − μ_c)` the first-order response of the
    /// negative subspace, which moves with `dA − μ_N dΦ` along the complement. The metric
    /// weight exists because in-band directions are priced at `Φ`'s own curvature: a value
    /// that moves with `Φ` there must be differentiated there.
    fn exact_hessian_basin_differential(
        block: &ExactHessianSpectralBlock,
        e_diag: &Array1<f64>,
        e_beta: Option<&Array2<f64>>,
        total_t: usize,
        basin: &ExactHessianBasin,
    ) -> Result<ExactHessianPricing, SaeCriterionError> {
        let dim = block.eigenvalues.len();
        let negative = &basin.negative;
        let complement = &basin.complement;
        let basis = &basin.basis;
        let q = negative.len();
        // The positive-inverse part `Σ wᵢwᵢᵀ/μᵢ` over the retained complement, as the
        // Gram `S·Sᵀ` of the columns `wᵢ/√μᵢ` (#2267). The rank-1 loop this replaces
        // paid `retained·dim²` scalar updates on every gradient evaluation.
        let retained: Vec<usize> = complement
            .iter()
            .copied()
            .filter(|&i| block.eigenvalues[i] > block.rank_floor(i))
            .collect();
        let scaled = Array2::from_shape_fn((dim, retained.len()), |(row, col)| {
            let i = retained[col];
            block.eigenvectors[[row, i]] / block.eigenvalues[i].sqrt()
        });
        let mut a_derivative = scaled.dot(&scaled.t());
        drop(scaled);
        let in_band: Vec<usize> = complement
            .iter()
            .copied()
            .filter(|&i| block.eigenvalues[i] <= block.rank_floor(i))
            .collect();
        let band_basis = Array2::from_shape_fn((dim, in_band.len()), |(row, col)| {
            block.eigenvectors[[row, in_band[col]]]
        });
        let mut metric_derivative = band_basis.dot(&band_basis.t());
        drop(band_basis);
        let mut clamp_diagonal_derivative = Array1::<f64>::zeros(total_t);
        let mut clamp_border_derivative = Array2::<f64>::zeros((dim - total_t, dim - total_t));
        let mut basin_inverse = Array2::<f64>::zeros((q, q));
        for (i, &inverse) in basin.inverse_values.iter().enumerate() {
            let vector = basin.vectors.column(i);
            if inverse == 0.0 {
                // An in-band basin direction is priced at the metric's curvature too.
                for row in 0..dim {
                    for col in 0..dim {
                        metric_derivative[[row, col]] += vector[row] * vector[col];
                    }
                }
                continue;
            }
            for row in total_t..dim {
                for col in total_t..dim {
                    clamp_border_derivative[[row - total_t, col - total_t]] += inverse * vector[row] * vector[col];
                }
            }
            for row in 0..dim {
                for col in 0..dim {
                    a_derivative[[row, col]] += inverse * vector[row] * vector[col];
                }
                if row < total_t {
                    clamp_diagonal_derivative[row] += inverse * vector[row] * vector[row];
                }
            }
            for row in 0..q {
                for col in 0..q {
                    basin_inverse[[row, col]] +=
                        inverse * basin.rotation[[row, i]] * basin.rotation[[col, i]];
                }
            }
        }
        if !negative.is_empty() && !complement.is_empty() {
            let other = Array2::from_shape_fn((dim, complement.len()), |(row, col)| {
                block.eigenvectors[[row, complement[col]]]
            });
            let mut e_cross = Array2::<f64>::zeros((q, complement.len()));
            for row in 0..total_t {
                for i in 0..q {
                    let left = e_diag[row] * basis[[row, i]];
                    for j in 0..complement.len() {
                        e_cross[[i, j]] += left * other[[row, j]];
                    }
                }
            }
            if e_beta.is_some() {
                // #2828 — `E` now has a border block, so the negative
                // projector's first-order response to it is part of `∂value/∂A`
                // too. Omitting it here would leave the value consistent and the
                // gradient not.
                let border = Self::dropped_curvature_border_forms(
                    e_beta,
                    total_t,
                    basis.view(),
                    other.view(),
                );
                for i in 0..q {
                    for j in 0..complement.len() {
                        e_cross[[i, j]] += border[[i, j]];
                    }
                }
            }
            let mut response = basin_inverse.dot(&e_cross);
            for (i, &negative_index) in negative.iter().enumerate() {
                for (j, &other_index) in complement.iter().enumerate() {
                    let gap = block.eigenvalues[negative_index] - block.eigenvalues[other_index];
                    if gap == 0.0 {
                        return Err(SaeCriterionError::Numerical(
                            "exact-A rank classification splits a repeated eigenspace; its negative projector is not differentiable"
                                .to_string(),
                        ));
                    }
                    response[[i, j]] /= gap;
                }
            }
            let cross = basis.dot(&response).dot(&other.t());
            // The negative subspace responds to `dA − μ_N dΦ`, so the metric carries the
            // same response weighted by each negative direction's curvature.
            let mut weighted_response = response;
            for (i, &negative_index) in negative.iter().enumerate() {
                weighted_response
                    .row_mut(i)
                    .mapv_inplace(|value| value * block.eigenvalues[negative_index]);
            }
            let metric_cross = basis.dot(&weighted_response).dot(&other.t());
            for row in 0..dim {
                for col in 0..dim {
                    a_derivative[[row, col]] += cross[[row, col]] + cross[[col, row]];
                    metric_derivative[[row, col]] -=
                        metric_cross[[row, col]] + metric_cross[[col, row]];
                }
            }
        }
        Ok(ExactHessianPricing {
            a_derivative,
            metric_derivative,
            clamp_diagonal_derivative,
            clamp_border_derivative,
        })
    }

    /// Materialize only the joint spectral geometry required by a dense
    /// exact-stationarity solve.  No log-determinant pricing or coordinate-block
    /// decomposition is paid on routes that rank the B majorizer.
    pub(crate) fn materialize_exact_stationarity_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<ExactHessianSpectralBlock, String> {
        let (a, _gap_border) =
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)?;
        Self::exact_hessian_spectral_block(a, &ArrowMetric::Joint(cache).prepare()?)
    }

    /// #2267 — materialize `A` at `cache` with the half of `E = B − A` its pricing reads,
    /// and decompose it once. The dense criterion prices `½log|A|` off the result and hands
    /// it to the evaluation, whose derivative reads the priced pseudo-inverse `A⁺` and the
    /// raw signed stationarity pseudoinverse off the SAME joint eigensystem and floor.
    pub(crate) fn materialize_dense_exact_a_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<DenseExactAGeometry, String> {
        let total_t = cache.delta_t_len();
        // #2828 — the border half of `E = B − A`, read off the same border probes
        // that build `A` (#2731).
        let (a, e_beta) =
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)?;
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let metric = ArrowMetric::Joint(cache).prepare()?;
        // #2234 — every closure-certified circle orbit alone in its connected block of `A` and `Φ`
        // is integrated exactly rather than priced by its chord curvature: the block prices the
        // stiffened `A_s`, and its solves eliminate the orbit coordinates to return `A⁺`, off this
        // one decomposition.
        let mut orbit_generators = Vec::new();
        for pricing in self.separated_compact_orbit_pricing(rho, target, cache)? {
            match pricing {
                CompactOrbitPricing::ExactCircle(generator) => {
                    log::info!(
                        "[SAE-EXACT-ORBIT] atom={} exact circle orbit: period={:e} eta={:e} \
                         closure residual={:.3e} band={:.3e} prior rows={}",
                        generator.atom,
                        generator.period,
                        generator.eta,
                        generator.closure_residual,
                        generator.closure_band,
                        generator.prior_rows.len(),
                    );
                    orbit_generators.push(generator);
                }
                CompactOrbitPricing::Laplace { atom, reason } => {
                    if reason != CompactOrbitLaplaceReason::NotAPeriodicChart {
                        log::info!("[SAE-EXACT-ORBIT] atom={atom} keeps Laplace pricing: {reason:?}");
                    }
                }
            }
        }
        let mut tangents = Array2::<f64>::zeros((a.nrows(), orbit_generators.len()));
        for (column, generator) in orbit_generators.iter().enumerate() {
            tangents.column_mut(column).assign(&generator.tangent);
        }
        let (operator, stiffening) = Self::stiffen_compact_orbits(a, tangents, &metric)?;
        let mut block = Self::exact_hessian_spectral_block(operator, &metric)?;
        if let Some(stiffening) = stiffening {
            block.orbit = Some(stiffening.eliminate(&block, &metric)?);
        }
        Ok(DenseExactAGeometry {
            block,
            e_diag,
            e_beta,
            total_t,
            orbit_generators,
            rank_charge_dispersion: None,
        })
    }

    /// `(c_k, [(t index, u_ik)])` per atom: the ordered Beta--Bernoulli prior's
    /// cross-row mass Hessian `Σ_k c_k u_k u_kᵀ` over the logit slots of `cache`'s
    /// coordinate layout, with `c_k = weight·d²L/dM²` and `u_ik = w_i dz_ik/dℓ_ik`.
    /// Empty for every other assignment mode. The dense materialization and the
    /// row-sandwich meat both read it, so they name one operator.
    pub(crate) fn ordered_mass_hessian_carriers(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<(f64, Vec<(usize, f64)>)>, String> {
        let Some(channels) = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            self.row_loss_weights.as_deref(),
        )?
        else {
            return Ok(Vec::new());
        };
        let offsets = &cache.row_offsets;
        let mut mass_carriers: Vec<(f64, Vec<(usize, f64)>)> = channels
            .mass_hessian_coefficient
            .iter()
            .map(|&coefficient| (coefficient, Vec::new()))
            .collect();
        for row in 0..cache.n_rows() {
            for (local, variable) in self.row_vars_for_cache_row(row, cache)?.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *variable {
                    let value = channels.z_jac[row * channels.k_max + atom];
                    if value != 0.0 {
                        mass_carriers[atom].1.push((offsets[row] + local, value));
                    }
                }
            }
        }
        Ok(mass_carriers)
    }

    /// #2330 — dense symmetric materialization of the EXACT stationarity
    /// Hessian `A = ∇²_θθ L = B + ΔC` (`dim×dim`, `dim = total_t + k`), built
    /// column by column via [`Self::apply_exact_hessian`] and symmetrized, at the
    /// small-dense (circle-mint) scale; shared by the observed-information
    /// log-determinant (VALUE) and its `A⁻¹` selected inverse (GRADIENT) so both
    /// factor one identical operator
    /// ([`Self::exact_observed_information_log_dets_with_saddle_directions`]).
    /// The exact stationarity Hessian as a dense `dim × dim` matrix, assembled
    /// from `slots + k` Hessian-vector applies instead of `dim` (gam#2267).
    ///
    /// The operator is an arrow plus the ordered-Beta--Bernoulli prior's
    /// cross-row mass rank-one blocks. Subtract those known low-rank actions
    /// from each batched probe and add their complete matrix once afterward.
    /// In the remaining arrow, each row couples only to itself and the border.
    /// Probing coordinate slot `c` of EVERY row at once
    /// therefore returns column `c` of every row's diagonal block in one apply
    /// (the cross-row `t–t` entries that sum would otherwise mix are zero), and
    /// probing border column `j` returns `A[·, β_j]` whole, its `t` entries
    /// included, so the `t–β` block comes from the `k` border probes by
    /// symmetry. On the #2267 K=8 arm (508 rows × 2 coordinates + 72) that is
    /// 74 applies where the column loop took 1088, 3.8 ms each, 129 times in
    /// the run's first 14 minutes.
    ///
    /// [`Self::materialize_exact_hessian_dense_by_columns`] is that column
    /// loop, kept as the oracle the equality pin measures this against.
    ///
    /// The border probes also yield `E_ββ`, the β-tier decoder-prior majorizer gap
    /// on the border (#2828): it is the negation of the leg-(5) columns those probes
    /// compute (#2731), symmetrized, and `None` when no β-tier prior is live and
    /// the block is identically zero.
    pub(crate) fn materialize_exact_hessian_dense_with_gap_border(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), String> {
        let dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
        let mut a = Array2::<f64>::zeros((dim, dim));
        let gap_border = self.probe_exact_hessian_arrow(rho, target, cache, &mut a)?;
        Ok((a, gap_border))
    }

    /// The probes of [`Self::materialize_exact_hessian_dense_with_gap_border`], written into
    /// `sink` (#2234): the dense route holds them as its `dim × dim` block and the arrow orbit
    /// lane as arrow blocks, the same entries either way. Returns the gap border.
    pub(crate) fn probe_exact_hessian_arrow<S: ExactHessianProbeSink + ?Sized>(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        sink: &mut S,
    ) -> Result<Option<Array2<f64>>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = sae_exact_stationarity_dim(total_t, k);
        let n_rows = cache.n_rows();
        let offsets = &cache.row_offsets;
        let mass_carriers = self.ordered_mass_hessian_carriers(rho, cache)?;
        let slots = (0..n_rows)
            .map(|row| offsets[row + 1] - offsets[row])
            .max()
            .unwrap_or(0);
        log::info!(
            "[SAE-EXACT-DENSE] materializing the exact stationarity Hessian: dim={dim} \
             (coords={total_t} + border={k}) from {} arrow probes ({slots} coordinate \
             slots + {k} border columns), {:.1} MiB per dim x dim f64 block",
            slots + k,
            sae_exact_stationarity_block_bytes(dim) as f64 / (1024.0 * 1024.0),
        );
        let build_started = std::time::Instant::now();
        // #2828 — one β-tier decoder-prior plan for all `slots + k` probes.
        let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
        // #2731 — and one residual-curvature plan: the row jets and residual are
        // contracted here once, where every probe used to rebuild them.
        let residual = self.prepare_residual_curvature_rows(target, cache)?;
        // #2731 — every probe is an independent apply of one fixed operator against
        // plans prepared once for this state, so the probes run on the rayon pool.
        // Columns are written serially in probe order: `a` is bit-identical to the
        // one-probe-at-a-time loop, and the first failing probe's error is the one
        // returned. Probes run in batches of the pool width, so at most one batch of
        // `dim`-length columns is held beside `a`. Job 391502 read 290 probes of
        // 699–827 ms each per polish step at `p = 2048, charts = 32`, on one core.
        use rayon::prelude::*;
        let pool_threads = rayon::current_num_threads().max(1);
        for batch_start in (0..slots).step_by(pool_threads) {
            let batch_end = (batch_start + pool_threads).min(slots);
            let columns: Vec<Result<SaeArrowVector, String>> = (batch_start..batch_end)
                .into_par_iter()
                .map(|slot| -> Result<SaeArrowVector, String> {
                    let mut unit = SaeArrowVector {
                        t: Array1::<f64>::zeros(total_t),
                        beta: Array1::<f64>::zeros(k),
                    };
                    for row in 0..n_rows {
                        let (start, end) = (offsets[row], offsets[row + 1]);
                        if start + slot < end {
                            unit.t[start + slot] = 1.0;
                        }
                    }
                    let mut av = self.apply_exact_hessian_prepared(
                        rho, cache, &unit, &prepared, &residual,
                    )?;
                    for (coefficient, carrier) in &mass_carriers {
                        let projection = carrier
                            .iter()
                            .map(|&(index, value)| value * unit.t[index])
                            .sum::<f64>();
                        for &(index, value) in carrier {
                            av.t[index] -= coefficient * value * projection;
                        }
                    }
                    Ok(av)
                })
                .collect();
            for (offset, av) in columns.into_iter().enumerate() {
                let av = av?;
                let slot = batch_start + offset;
                for row in 0..n_rows {
                    let (start, end) = (offsets[row], offsets[row + 1]);
                    if start + slot < end {
                        sink.row_slot_column(start, end, slot, &av.t);
                    }
                }
            }
        }
        sink.add_mass_carriers(&mass_carriers)?;
        // #2731 — each border probe is `apply_exact_hessian_prepared` with leg (5)
        // of `ΔC` computed here and folded into `ΔC·e_j` exactly as
        // `apply_exact_hessian_minus_b_prepared` folds it, so the column is
        // bit-identical, and the leg column is kept: `E_ββ` is its negation, so the
        // gap border needs no second pass of `k` β-tier applies. Job 539190 read that
        // second pass at 11.109–13.432 s of each 32.00–40.34 s polish step
        // (`p = 2048, charts = 32, k = 288`).
        let projection = crate::frames::FrameProjection::new(self);
        let mut leg_columns = Array2::<f64>::zeros((k, k));
        for batch_start in (0..k).step_by(pool_threads) {
            let batch_end = (batch_start + pool_threads).min(k);
            let columns: Vec<Result<(SaeArrowVector, Array1<f64>), String>> =
                (batch_start..batch_end)
                    .into_par_iter()
                    .map(|j| -> Result<(SaeArrowVector, Array1<f64>), String> {
                        let mut unit = SaeArrowVector {
                            t: Array1::<f64>::zeros(total_t),
                            beta: Array1::<f64>::zeros(k),
                        };
                        unit.beta[j] = 1.0;
                        let b_v =
                            apply_raw_cached_arrow_hessian(cache, unit.t.view(), unit.beta.view())?;
                        let mut dc_v = self
                            .apply_exact_hessian_minus_b_prepared_before_beta_prior_leg(
                                rho, cache, &unit, &residual,
                            )?;
                        let leg = self.decoder_prior_gap_border_leg(
                            cache,
                            &prepared,
                            &projection,
                            unit.beta.view(),
                        )?;
                        for (index, &value) in leg.iter().enumerate() {
                            dc_v.beta[index] += value;
                        }
                        let av = SaeArrowVector {
                            t: &b_v.t + &dc_v.t,
                            beta: &b_v.beta + &dc_v.beta,
                        };
                        Ok((av, leg))
                    })
                    .collect();
            for (offset, column) in columns.into_iter().enumerate() {
                let (av, leg) = column?;
                let j = batch_start + offset;
                sink.border_column(total_t, j, &av);
                for i in 0..k {
                    leg_columns[[i, j]] = leg[i];
                }
            }
        }
        sink.symmetrize();
        // `E = B − A` and leg (5) is the β-tier part of `A − B` on the border, so
        // `E_ββ` negates the kept columns. The remainder is a difference of two
        // symmetric operators; symmetrize the probe assembly so the basin quadratic
        // forms cannot pick up an asymmetric round-off residue.
        let gap_border = if k == 0 {
            None
        } else {
            let mut gap = leg_columns.mapv(|value| -value);
            if gap.iter().all(|&value| value == 0.0) {
                None
            } else {
                for row in 0..k {
                    for col in (row + 1)..k {
                        let average = 0.5 * (gap[[row, col]] + gap[[col, row]]);
                        gap[[row, col]] = average;
                        gap[[col, row]] = average;
                    }
                }
                Some(gap)
            }
        };
        let build_elapsed = build_started.elapsed();
        log::info!(
            "[SAE-EXACT-DENSE] operator BUILT: dim={dim}, {} arrow probes on {pool_threads} pool \
             threads + symmetrization in {:.3} s ({:.3} ms wall per probe), with the \
             decoder-prior majorizer gap border from the same {k} border probes; the \
             O(dim^3) symmetric eigendecomposition has NOT started yet",
            slots + k,
            build_elapsed.as_secs_f64(),
            build_elapsed.as_secs_f64() * 1.0e3 / ((slots + k).max(1) as f64),
        );
        Ok(gap_border)
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
    /// concave half, 0 elsewhere). ThresholdGate logits carry the derivative of
    /// their own `majorized - exact` remainder from the shared curvature seam.
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
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        for row in 0..self.n_obs() {
            let base = cache.row_offsets[row];
            let vars = self.row_vars_for_cache_row(row, cache)?;
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in vars.iter().enumerate() {
                if let SaeLocalRowVar::Logit { atom } = *va {
                    if self.assignment.logits_are_fixed() {
                        continue;
                    }
                    if let AssignmentMode::ThresholdGate {
                        temperature,
                        threshold,
                    } = self.assignment.mode
                    {
                        let curvature = crate::assignment::ThresholdGateLogitCurvature::eval(
                            w_row * rho.lambda_sparse()?,
                            self.assignment.logits[[row, atom]],
                            threshold,
                            1.0 / temperature,
                        );
                        dt[base + a] = curvature.majorized_hess_logit_derivative()
                            - curvature.exact_hess_logit_derivative();
                    }
                    continue;
                }
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

    /// Chain the diagonal E derivative of one priced block to the model's
    /// strength and theta coordinates. Both results use full logdet units.
    fn priced_clamp_adjoint_extras(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        pricing: &ExactHessianPricing,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let total_t = cache.delta_t_len();
        if pricing.clamp_diagonal_derivative.len() != total_t {
            return Err("priced clamp derivative has the wrong coordinate dimension".to_string());
        }
        let e_diag = self.materialize_ard_concave_clamp_diagonal(rho, cache)?;
        let de_dt = self.ard_concave_clamp_dt_diagonal(rho, cache)?;
        let coord_axis = self.coord_axis_map_for_cache(cache)?;
        let mut strength_flat: Vec<Option<usize>> = coord_axis
            .iter()
            .map(|axis| axis.map(|(atom, axis)| rho.ard_flat_index(atom, axis)))
            .collect();
        if matches!(self.assignment.mode, AssignmentMode::ThresholdGate { .. }) {
            for row in 0..self.n_obs() {
                let variables = self.row_vars_for_cache_row(row, cache)?;
                for (local, variable) in variables.iter().enumerate() {
                    let slot = cache.row_offsets[row] + local;
                    if matches!(variable, SaeLocalRowVar::Logit { .. }) && e_diag[slot] != 0.0 {
                        strength_flat[slot] = Some(rho.sparse_flat_index().ok_or_else(|| {
                            "nonzero threshold-gate clamp has no sparse strength coordinate"
                                .to_string()
                        })?);
                    }
                }
            }
        }
        let mut delta_trace = Array1::<f64>::zeros(rho.flat_coordinates().len());
        let mut delta_gamma_t = Array1::<f64>::zeros(total_t);
        for r in 0..total_t {
            let weight = pricing.clamp_diagonal_derivative[r];
            if let Some(flat) = strength_flat[r] {
                let contribution = weight * e_diag[r];
                if contribution != 0.0 {
                    delta_trace[flat] += contribution;
                }
            }
            delta_gamma_t[r] = weight * de_dt[r];
        }
        Ok((delta_trace, delta_gamma_t))
    }

    pub(crate) fn dense_exact_a_logdet_channels(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        geometry: &DenseExactAGeometry,
        rank_charge_theta: &SaeArrowVector,
    ) -> Result<DenseExactALogdetChannels, String> {
        let n_params = rho.flat_coordinates().len();
        // #2267 — the channels price the block the evaluation's value was classified on, so
        // that block must have been materialized at this cache.
        let dim = sae_exact_stationarity_dim(cache.delta_t_len(), cache.k);
        if geometry.total_t != cache.delta_t_len() || geometry.block.eigenvalues.len() != dim {
            return Err(format!(
                "dense_exact_a_logdet_channels: the spectral block (dim {}, {} coordinate slots) \
                 was not materialized at this cache (dim {dim}, {} coordinate slots)",
                geometry.block.eigenvalues.len(),
                geometry.total_t,
                cache.delta_t_len(),
            ));
        }
        let pricing = Self::price_exact_hessian_block(
            &geometry.block,
            &geometry.e_diag,
            geometry.e_beta.as_ref(),
            geometry.total_t,
            "joint",
        )
        .map_err(|error| error.to_string())?;
        // The common basin owner includes the negative-subspace response in
        // dA. Chain its remaining explicit dE term to rho and theta here.
        let (priced_joint_trace, priced_joint_gamma) =
            self.priced_clamp_adjoint_extras(rho, cache, &pricing)?;
        // #2234 — an orbit-stiffened block prices `log|A_s| − log det N − 2·Σ log I_k + K·log 2π`.
        // Its differential replaces the block's own `dA` and `dΦ` weights and adds the legs that
        // reach neither operator: each orbit integral's coordinate and log-precision legs and each
        // tangent's border legs.
        let (pricing, orbit_legs) = if geometry.orbit_generators.is_empty() {
            (pricing, None)
        } else {
            let differential = Self::compact_orbit_differential(
                &geometry.orbit_generators,
                &geometry.block,
                &pricing,
                &ArrowMetric::Joint(cache).prepare()?,
                geometry.total_t,
            )?;
            (
                ExactHessianPricing {
                    a_derivative: differential.operator_weight,
                    metric_derivative: differential.metric_weight,
                    clamp_diagonal_derivative: pricing.clamp_diagonal_derivative,
                    clamp_border_derivative: pricing.clamp_border_derivative,
                },
                Some((differential.theta, differential.log_precisions)),
            )
        };
        let a_pinv = &pricing.a_derivative;
        // This value diagonalizes `A_raw = B_raw + ΔC`; differentiate that raw
        // operator, not the row-conditioned operator carried by arrow factors.
        let da_by_flat = self.exact_stationarity_penalty_derivatives_by_flat(rho, cache)?;
        let frob = |x: &Array2<f64>, y: &Array2<f64>| -> f64 { (x * y).sum() };
        let mut logdet_trace = Array1::<f64>::zeros(n_params);
        for (&i, da) in da_by_flat.iter() {
            logdet_trace[i] = 0.5 * frob(&a_pinv, da);
        }
        // One per-coordinate map at a time: the metric channel below builds its own.
        drop(da_by_flat);
        // Ordered-Beta–Bernoulli sparse coordinate: its ∂A/∂ρ_sparse is the exact
        // integrated-marginal logit Hessian (cross-row), absent from the operator
        // map above (softmax-only). Add its ½log|A| trace directly.
        if let Some(sparse) = rho.sparse_flat_index() {
            if matches!(
                self.assignment.mode,
                AssignmentMode::OrderedBetaBernoulli { .. }
            ) {
                logdet_trace[sparse] =
                    self.dense_exact_a_ordered_bb_sparse_trace(rho, cache, &a_pinv)?;
            }
        }
        let mut gamma = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &a_pinv,
            true,
            true,
            Some(target),
        )?;
        // The scalar criterion carries a leading half; Gamma is the full
        // log-determinant theta derivative used in the caller's -1/2 IFT fold.
        logdet_trace.scaled_add(0.5, &priced_joint_trace);
        gamma.t += &priced_joint_gamma;
        gamma.beta += &self.decoder_prior_gap_theta_trace(
            cache, pricing.clamp_border_derivative.view(),
        )?;
        // #2933 F07 — in-band pencil directions are priced at `Φ`'s own curvature, so the
        // value moves with the evidence factor there as well.
        let (metric_trace, metric_gamma) = self.evidence_metric_derivative_channels(
            rho,
            target,
            cache,
            &pricing.metric_derivative,
        )?;
        logdet_trace += &metric_trace;
        gamma.t += &metric_gamma.t;
        gamma.beta += &metric_gamma.beta;
        // #2234 — the orbit legs that reach neither `A` nor `Φ`.
        if let Some((theta, log_precisions)) = orbit_legs {
            gamma.t += &theta.t;
            gamma.beta += &theta.beta;
            for (atom, log_precision) in log_precisions {
                if !rho.log_ard[atom].is_empty() {
                    logdet_trace[rho.ard_flat_index(atom, 0)] += 0.5 * log_precision;
                }
            }
        }
        // #2267 — the caller's rank-charge derivative, read off the same block.
        gamma.t.scaled_add(2.0, &rank_charge_theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge_theta.beta);
        let stationarity_adjoint = geometry.block.solve_stationarity(&gamma)?.step;
        Ok(DenseExactALogdetChannels {
            logdet_trace,
            theta_adjoint: gamma,
            stationarity_adjoint,
        })
    }

    /// #2933 F07 — `(½⟨X, ∂Φ/∂ρ⟩, ⟨X, ∂Φ/∂θ⟩)` for a dense symmetric weight `X` on the
    /// joint `(t, β)` layout, where `Φ = Φ(B_raw)` is the conditioned evidence factor the
    /// exact-`A` pencil is classified and priced in. The trace half carries the
    /// criterion's leading `½`; the θ half is in full log-determinant units, like every
    /// θ-adjoint here.
    ///
    /// [`Self::evidence_metric_raw_weight`] folds both conditionings of `Φ` into a weight
    /// on `dB_raw`, which the raw-majorizer builders contract: the per-coordinate
    /// curvature operators, the ordered Beta--Bernoulli majorized diagonal, and the θ legs
    /// of [`Self::logdet_theta_adjoint_dense`] with the decoder priors' border and the
    /// ordered Beta--Bernoulli shared-mass leg added, as the majorizer channels add them.
    /// `target` reaches that tower's embedded-sphere rows, whose Riemannian conversion
    /// reads the row residual (#2933 F24).
    pub(crate) fn evidence_metric_derivative_channels(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        weight: &Array2<f64>,
    ) -> Result<(Array1<f64>, SaeArrowVector), String> {
        let total_t = cache.delta_t_len();
        let mut trace = Array1::<f64>::zeros(rho.flat_coordinates().len());
        if weight.iter().all(|&value| value == 0.0) {
            return Ok((
                trace,
                SaeArrowVector {
                    t: Array1::zeros(total_t),
                    beta: Array1::zeros(cache.k),
                },
            ));
        }
        let raw_weight = self.evidence_metric_raw_weight(cache, weight)?;
        for (flat, operator) in self.raw_penalty_curvature_operators_by_flat(rho, cache)? {
            trace[flat] = 0.5 * (&raw_weight * &operator).sum();
        }
        let row_weights = self.row_loss_weights.as_deref();
        let ordered_channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        let k_atoms = self.k_atoms();
        // `(flat logit slot, joint index)` of every free ordered Beta--Bernoulli logit.
        let mut logit_sites: Vec<(usize, usize)> = Vec::new();
        if let Some(channels) = ordered_channels.as_ref() {
            for row in 0..self.n_obs() {
                for (local, variable) in self.row_vars_for_cache_row(row, cache)?.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *variable {
                        if atom < k_atoms {
                            logit_sites.push((row * k_atoms + atom, cache.row_offsets[row] + local));
                        }
                    }
                }
            }
            // The sparse coordinate has no assembled operator here
            // (`SparseLogitCurvature::CrossRowOwnedElsewhere`): the majorizer is the positive
            // part of the row-local diagonal, whose log-strength derivative the majorizer
            // trace reads slot by slot.
            // A fixed concentration puts no coordinate into the prior (#2933 F45), so the
            // majorizer does not move with `ρ_sparse` there.
            if let Some(sparse) = rho
                .sparse_flat_index()
                .filter(|_| self.assignment.effective_alpha_is_learnable())
            {
                let hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
                    &self.assignment,
                    rho,
                    row_weights,
                )?;
                if !hdiag.is_empty() {
                    let mut accumulated = 0.0_f64;
                    for &(slot, global) in &logit_sites {
                        let (row, atom) = (slot / k_atoms, slot % k_atoms);
                        let majorized = super::construction_arrow_schur_assembly::ordered_beta_bernoulli_psd_majorized_log_alpha_hdiag(
                            channels, row, k_atoms, atom, hdiag[slot],
                        );
                        accumulated += raw_weight[[global, global]] * majorized;
                    }
                    trace[sparse] = 0.5 * accumulated;
                }
            }
        }
        // θ: both conditionings are already folded into `raw_weight`, so the dense
        // adjoint's own row Daleckii--Krein correction is skipped.
        let mut gamma =
            self.logdet_theta_adjoint_dense(rho, cache, &raw_weight, true, false, Some(target))?;
        if cache.k > 0 {
            // `B_ββ` carries the decoder priors' majorizer, `A_ββ + E_ββ`.
            let border = raw_weight.slice(s![total_t.., total_t..]);
            gamma.beta += &self.exact_decoder_prior_theta_trace(cache, border)?;
            gamma.beta += &self.decoder_prior_gap_theta_trace(cache, border)?;
        }
        if let Some(channels) = ordered_channels.as_ref() {
            // `Φ` carries `B`'s ordered Beta--Bernoulli prior diagonal, the positive part of its
            // row-local term. It moves with its own logit (`local_logit_third`) and with every
            // logit of its column through the shared mass (`m_channel`, `∂M/∂ℓ = z_jac`). The
            // dense majorizer builder carries no ordered prior channel on this route, so both
            // legs are contracted here, as `logdet_theta_adjoint_from_probes` contracts them.
            let mut column_coefficient = vec![0.0_f64; k_atoms];
            for &(slot, global) in &logit_sites {
                column_coefficient[slot % k_atoms] +=
                    raw_weight[[global, global]] * channels.m_channel[slot];
            }
            for &(slot, global) in &logit_sites {
                gamma.t[global] += raw_weight[[global, global]] * channels.local_logit_third[slot]
                    + column_coefficient[slot % k_atoms] * channels.z_jac[slot];
            }
        }
        Ok((trace, gamma))
    }

    /// #2933 F07 — the weight `X̃` on `dB_raw` with `⟨X̃, dB_raw⟩ = ⟨X, dΦ⟩`, for the
    /// conditioned evidence factor `Φ = Φ(B_raw)` this cache represents.
    ///
    /// ```text
    ///   Φ = [ T     C           ]    T = ⊕ᵢ φᵢ(B_raw,tt⁽ⁱ⁾) + VᵢVᵢᵀ
    ///       [ Cᵀ    S̃ + CᵀT⁻¹C  ]    S̃ = φ_S(P S P + QQᵀ),   S = D − CᵀT⁻¹C
    /// ```
    ///
    /// with `C = B_raw,tβ` and `D = B_raw,ββ` raw, `φᵢ`/`φ_S` the spectral pin maps the
    /// cache records, `Vᵢ` the structural row gauge pins and `P = I − QQᵀ` the border gauge
    /// quotient; both gauge pins are constant. A Daleckii--Krein map
    /// `Dφ[dM] = U(F ∘ UᵀdMU)Uᵀ` is self-adjoint, so with `Z = P·Dφ_S[X_ββ]·P`,
    /// `Y = X_ββ − Z` and `G = T⁻¹C`,
    ///
    /// ```text
    ///   ⟨X, dΦ⟩ = Σᵢ⟨Dφᵢ[(X_tt − GYGᵀ)ᵢ], dB_raw,tt⁽ⁱ⁾⟩ + 2⟨X_tβ + GY, dC⟩ + ⟨Z, dD⟩.
    /// ```
    ///
    /// Cross-row entries of `X̃_tt` are left in place: every raw curvature derivative is
    /// row-local there, so they contract nothing.
    fn evidence_metric_raw_weight(
        &self,
        cache: &ArrowFactorCache,
        weight: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let total_t = cache.delta_t_len();
        let k = cache.k;
        if weight.dim() != (total_t + k, total_t + k) {
            return Err(format!(
                "evidence_metric_raw_weight: weight {:?} on joint dimension {}",
                weight.dim(),
                total_t + k
            ));
        }
        let mut out = weight.clone();
        if k > 0
            && (cache.beta_schur_conditioning.is_some() || cache.beta_gauge_quotient.is_some())
        {
            let border = weight.slice(s![total_t.., total_t..]).to_owned();
            let mut schur_weight = match cache.beta_schur_conditioning.as_ref() {
                Some(spectrum) => Self::beta_schur_conditioning_fold(spectrum, &border)?,
                None => border.clone(),
            };
            if let Some(quotient) = cache.beta_gauge_quotient.as_ref() {
                // `X → (I − qqᵀ)X(I − qqᵀ)`, one orthonormal direction at a time.
                for direction in quotient.directions.iter() {
                    let image = schur_weight.dot(direction);
                    let along = direction.dot(&image);
                    for a in 0..k {
                        for b in 0..k {
                            schur_weight[[a, b]] += along * direction[a] * direction[b]
                                - direction[a] * image[b]
                                - image[a] * direction[b];
                        }
                    }
                }
            }
            let remainder = &border - &schur_weight;
            if remainder.iter().any(|&value| value != 0.0) {
                let mut unit = Array1::<f64>::zeros(k);
                for row in 0..cache.n_rows() {
                    let q = cache.row_dims[row];
                    let base = cache.row_offsets[row];
                    let factor = cache.undamped_factor(row);
                    // `Gᵢ = Tᵢ⁻¹Cᵢ`, one border column at a time.
                    let mut graph = Array2::<f64>::zeros((q, k));
                    for column in 0..k {
                        unit[column] = 1.0;
                        let mut coupled = Array1::<f64>::zeros(q);
                        let applied = cache.apply_htbeta_row(row, unit.view(), &mut coupled);
                        unit[column] = 0.0;
                        if !applied {
                            return Err(format!(
                                "evidence_metric_raw_weight: H_tβ^({row}) apply failed"
                            ));
                        }
                        graph
                            .column_mut(column)
                            .assign(&cholesky_solve_vector(factor, coupled.view()));
                    }
                    let graph_remainder = graph.dot(&remainder);
                    let within = graph_remainder.dot(&graph.t());
                    for a in 0..q {
                        for b in 0..q {
                            out[[base + a, base + b]] -= within[[a, b]];
                        }
                        for c in 0..k {
                            out[[base + a, total_t + c]] += graph_remainder[[a, c]];
                            out[[total_t + c, base + a]] += graph_remainder[[a, c]];
                        }
                    }
                }
            }
            out.slice_mut(s![total_t.., total_t..]).assign(&schur_weight);
        }
        for row in 0..cache.n_rows() {
            let Some(spectrum) = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            if spectrum.evecs.dim() != (q, q) {
                return Err(format!(
                    "evidence_metric_raw_weight: row {row} has dimension {q}, but its spectral \
                     carrier is {:?}",
                    spectrum.evecs.dim()
                ));
            }
            let block = out.slice(s![base..base + q, base..base + q]).to_owned();
            let folded = Self::deflation_folded_trace_weight(&block, &[], Some(spectrum));
            out.slice_mut(s![base..base + q, base..base + q]).assign(&folded);
        }
        Ok(out)
    }

    /// `Dφ_S[X] = Q(F ∘ QᵀXQ)Qᵀ` for the reduced-Schur spectral conditioning the evidence
    /// factor recorded, with the gap convention of
    /// [`Self::row_deflation_frechet_coefficients`]: a unit pin has `φ' = 0` and a raw
    /// direction `φ' = 1`. A clamp-basin price moves with `E`, which this map does not
    /// carry, so a spectrum holding one is refused rather than differentiated as though
    /// its price were constant.
    fn beta_schur_conditioning_fold(
        spectrum: &gam_solve::arrow_schur::BetaSchurConditioningSpectrum,
        weight: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        use gam_solve::arrow_schur::BetaSchurSpectralConditioning;
        let k = weight.nrows();
        if spectrum.evecs.dim() != (k, k)
            || spectrum.raw_evals.len() != k
            || spectrum.cond_evals.len() != k
            || spectrum.conditioning.len() != k
        {
            return Err(format!(
                "evidence_metric_raw_weight: the recorded reduced-Schur spectrum does not match \
                 border width {k}"
            ));
        }
        if spectrum
            .conditioning
            .iter()
            .any(|branch| *branch == BetaSchurSpectralConditioning::ClampBasin)
        {
            return Err(
                "evidence_metric_raw_weight: the reduced-Schur spectrum prices a clamp basin, \
                 whose price moves with E; the pencil metric derivative there is not modelled"
                    .to_string(),
            );
        }
        let raw = &spectrum.raw_evals;
        let conditioned = &spectrum.cond_evals;
        let eigen_scale = raw
            .iter()
            .chain(conditioned.iter())
            .copied()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let gap_threshold = eigen_gap_threshold(eigen_scale, k);
        let mut folded = spectrum.evecs.t().dot(weight).dot(&spectrum.evecs);
        for a in 0..k {
            for b in 0..k {
                let denominator = raw[a] - raw[b];
                let coefficient = if denominator.abs() > gap_threshold {
                    (conditioned[a] - conditioned[b]) / denominator
                } else if spectrum.conditioning[a] == BetaSchurSpectralConditioning::Raw {
                    1.0
                } else {
                    0.0
                };
                folded[[a, b]] *= coefficient;
            }
        }
        Ok(spectrum.evecs.dot(&folded).dot(&spectrum.evecs.t()))
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
    /// `Σ_{i,j} inv[i,j]·∂H_obb[i,j]/∂ℓ_w`, the full prior contribution the
    /// reconstruction and softmax-only row loops cannot carry. Per column `c`,
    /// `H_obb = weight·S'_c·uuᵀ + diag(D_i)` with `u_i = w_i·z_i(1−z_i)/τ`,
    /// `curv_i = z_i(1−z_i)(1−2z_i)/τ²`, `D_i = weight·S_c·w_i·curv_i`. Its logit
    /// derivative contracts to (with `P = uᵀ inv_cc u`, `(inv·u)_r`,
    /// `G = Σ_i inv[i,i]·w_i·curv_i`, `curv'_i = z_i(1−z_i)(1−6z_i+6z_i²)/τ³`):
    ///   `Γ[w=(r,c)] = weight·{ S''_c·u_r·P + 2·S'_c·w_r·curv_r·(inv·u)_r
    ///                          + S'_c·u_r·G + S_c·inv[r,r]·w_r·curv'_r }`.
    /// Contracts the matrix the caller passes as `inv` (production passes `A⁺`
    /// of the joint operator), on the logit t-slots.
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
        // Structural quantities per (row, column): (u, curv, curv', w).
        let uval = |row: usize, col: usize| -> (f64, f64, f64, f64) {
            let z = data.z[row * k + col];
            let w = data.row_weight[row];
            let zc = z * (1.0 - z);
            let u = w * zc * inv_tau;
            let curv = zc * (1.0 - 2.0 * z) * inv_tau2;
            let curvp = zc * (1.0 - 6.0 * z + 6.0 * z * z) * inv_tau3;
            (u, curv, curvp, w)
        };
        for col in 0..k {
            let s = data.score[col];
            let sp = data.score_derivative[col];
            let spp = data.score_second[col];
            let rows: Vec<usize> = (0..n).filter(|&r| gindex[r][col].is_some()).collect();
            let mut au = vec![0.0_f64; n];
            let mut p = 0.0_f64;
            let mut g = 0.0_f64;
            for &ri in &rows {
                let gi = gindex[ri][col].expect("row filtered to Some");
                let (ui, curvi, _curvpi, wi) = uval(ri, col);
                let mut au_ri = 0.0_f64;
                for &rj in &rows {
                    let gj = gindex[rj][col].expect("row filtered to Some");
                    let (uj, _, _, _) = uval(rj, col);
                    au_ri += inv[[gi, gj]] * uj;
                }
                au[ri] = au_ri;
                p += ui * au_ri;
                g += inv[[gi, gi]] * wi * curvi;
            }
            for &ri in &rows {
                let gi = gindex[ri][col].expect("row filtered to Some");
                let (ui, curvi, curvpi, wi) = uval(ri, col);
                let val = spp * ui * p
                    + 2.0 * sp * wi * curvi * au[ri]
                    + sp * ui * g
                    + s * inv[[gi, gi]] * wi * curvpi;
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
    ) -> Result<f64, String> {
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
        let hdiag = crate::assignment::assignment_prior_log_strength_hdiag_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        if hdiag.is_empty() {
            // Inert / frozen prior: ∂B and ΔC are both zero.
            return Ok(0.0);
        }
        if !self.assignment.effective_alpha_is_learnable() {
            // A fixed concentration puts no coordinate into the prior (#2933 F45), so
            // `∂A/∂ρ_sparse` is zero on the logit block.
            return Ok(0.0);
        }
        let channels = ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
            &self.assignment,
            rho,
            row_weights,
        )?;
        // A learnable concentration makes `ρ_sparse = log(α/α_base)` move only the
        // Beta shapes `a_k` (the prior weight stays one), so on the logit block
        // `∂A/∂ρ_sparse` is the concentration derivative of the EXACT prior Hessian:
        // `B` and `ΔC` split one operator and their majorizer parts cancel. Per atom
        // column that derivative is `∂S'_k/∂ρ·u uᵀ + diag(∂S_k/∂ρ·w_i·curv_i)` with
        // `u = w·J` (`z_jac`), and `hdiag` already holds its full diagonal
        // `∂S'_k/∂ρ·u_i² + ∂S_k/∂ρ·w_i·curv_i`. The trace is one rank-one quadratic
        // form per column plus the row-local part of the diagonal.
        let ch = channels.as_ref().ok_or_else(|| {
            "dense_exact_a_ordered_bb_sparse_trace: a learnable concentration needs the \
             ordered Beta--Bernoulli prior channels"
                .to_string()
        })?;
        let mut tr_joint = 0.0_f64;
        for atom in 0..k_atoms {
            let mass_curvature = ch.mass_hessian_log_alpha_derivative[atom];
            let mut quadratic = 0.0_f64;
            for irow in 0..n {
                let Some(gi) = logit_gindex[irow][atom] else {
                    continue;
                };
                let islot = irow * k_atoms + atom;
                let ui = ch.z_jac[islot];
                tr_joint += a_pinv[[gi, gi]] * (hdiag[islot] - mass_curvature * ui * ui);
                for jrow in 0..n {
                    let Some(gj) = logit_gindex[jrow][atom] else {
                        continue;
                    };
                    quadratic += a_pinv[[gi, gj]] * ui * ch.z_jac[jrow * k_atoms + atom];
                }
            }
            tr_joint += mass_curvature * quadratic;
        }
        Ok(0.5 * tr_joint)
    }

    /// Assemble `ΔC = A − B` per row, so the arrow evidence system can carry the
    /// EXACT observed information instead of the Newton/Schur majorizer.
    ///
    /// `Self::apply_exact_hessian_minus_b` contracts these blocks against a
    /// direction without ever forming them, which is all a matvec consumer needs.
    /// The streaming log-determinant is not a matvec consumer: it takes
    /// `log|H_tt^(i)|` off assembled per-row factors and reduces an assembled
    /// border, so it can only price `A` if the blocks exist (#2509).
    ///
    /// Every channel is row-local — (1a)/(1b) residual curvature, (2) the softmax
    /// entropy-minus-Gershgorin delta, (3) the periodic ARD concave clamp, (3b) the
    /// ThresholdGate concave remainder on logit slots — except
    /// ordered Beta–Bernoulli, whose integrated-marginal prior couples every row
    /// within an atom column. That mode has no arrow-structured `ΔC` and is
    /// REFUSED here rather than silently dropped: pricing `B` while claiming `A`
    /// is the defect this function exists to remove.
    ///
    /// These are the row and cross-block legs only. Leg (5), `ΔC_ββ`, lives on the
    /// border, and `Self::exact_a_evidence_system` composes it into the shared
    /// block (#2828).
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
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
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
        // (3b) #2520 — the ThresholdGate's concave remainder, from the producer the
        // applier and the clamp diagonal read; `None` off the threshold gate.
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

        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        let mut assignments = Array1::<f64>::zeros(k_atoms);

        let sphere_tangents = self.sphere_tangent_blocks(row_dims)?;
        let mut next_sphere_block = 0usize;
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
            // (3b) #2520 threshold gate: the applier's channel (3b), on logit slots.
            // `B` carries the PSD clamp of the gate's curvature. Without the
            // non-positive remainder the arrow system prices `B` on every switched-on
            // logit, while the classification's clamp diagonal restores a concave
            // half the operator never subtracted (#2915). The producer already
            // applies `w_row` and the fixed-logit mask.
            if let Some(remainder) = threshold_gate_remainder.as_ref() {
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        tt[[a, a]] += neg;
                    }
                }
            }
            // #2933 F36 — a sphere block's `ΔC_tt` is `P·ΔC_tt·P` and its `ΔC_tβ` is
            // `P·ΔC_tβ`, in the tangent projector `B`'s row was assembled in.
            while let Some(block) = sphere_tangents
                .get(next_sphere_block)
                .filter(|block| block.row == row)
            {
                for mut column in tt.axis_iter_mut(ndarray::Axis(1)) {
                    block.project_local(&mut column);
                }
                for mut tt_row in tt.axis_iter_mut(ndarray::Axis(0)) {
                    block.project_local(&mut tt_row);
                }
                for mut column in tbeta.axis_iter_mut(ndarray::Axis(1)) {
                    block.project_local(&mut column);
                }
                next_sphere_block += 1;
            }

            rows_out.push(ExactHessianDeltaRow { tt, tbeta });
        }
        Ok(rows_out)
    }
}

#[cfg(test)]
mod test_support {
    use super::Side;
    use super::{
        ArrowFactorCache, DeflatedArrowSolver, SaeArrowVector, SaeManifoldRho,
    };
    use gam_linalg::faer_ndarray::FaerEigh;
    use ndarray::{Array1, Array2};

    /// A block in the identity metric, where the pencil is the ordinary spectrum.
    fn spectral_fixture(a: &ndarray::Array2<f64>) -> super::ExactHessianSpectralBlock {
        let identity = Array2::<f64>::eye(a.nrows());
        let metric =
            super::tests_pencil_classification_2933::DensePencilMetric::new(identity.clone(), &identity)
                .expect("identity metric");
        super::SaeManifoldTerm::exact_hessian_spectral_block(a.clone(), &metric)
            .expect("symmetric fixture")
    }

    struct PricedFixture {
        log_det: f64,
        a_derivative: ndarray::Array2<f64>,
        clamp_diagonal_derivative: Array1<f64>,
    }

    #[test]
    fn ordered_bb_batched_exact_hessian_preserves_cross_row_mass_curvature_2820() {
        for weighted in [false, true] {
            let (mut term, target, rho) =
                crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, -1.0);
            if weighted {
                term.set_row_loss_weights(
                    (0..term.n_obs())
                        .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                        .collect(),
                )
                .expect("positive design weights");
            }
            // Operator equality applies at indefinite states too. Build the
            // positive Newton metric directly, without asking the evidence
            // criterion to admit this deliberately nonstationary state.
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("fixed-state OBB assembly");
            let (_, _, cache) = super::solve_arrow_newton_step_with_options(
                &system,
                1.0e-6,
                1.0e-6,
                &super::ArrowSolveOptions::direct(),
            )
            .expect("positive Newton metric");
            let oracle = term
                .materialize_exact_hessian_dense_by_columns(&rho, target.view(), &cache)
                .expect("independent column-probe oracle");
            let batched = term
                .materialize_exact_hessian_dense(&rho, target.view(), &cache)
                .expect("arrow plus mass assembly");
            let mut cross_row_signal = 0.0_f64;
            for row in 0..cache.n_rows() {
                for i in cache.row_offsets[row]..cache.row_offsets[row + 1] {
                    for j in cache.row_offsets[row + 1]..cache.delta_t_len() {
                        cross_row_signal = cross_row_signal.max(oracle[[i, j]].abs());
                    }
                }
            }
            assert!(
                cross_row_signal > 1.0e-4,
                "cross-row mass block must be live"
            );
            let scale = oracle
                .iter()
                .map(|value| value.abs())
                .fold(1.0_f64, f64::max);
            let error = (&batched - &oracle)
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error <= 1.0e-12 * scale,
                "weighted={weighted}: batched/column error={error:e}, scale={scale:e}"
            );
        }
    }

    #[test]
    fn ordered_bb_full_prior_theta_adjoint_matches_existing_hvp_difference_2820() {
        for weighted in [false, true] {
            let (mut term, target, rho) =
                crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, -1.0);
            if weighted {
                term.set_row_loss_weights(
                    (0..term.n_obs())
                        .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                        .collect(),
                )
                .expect("positive design weights");
            }
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("OBB assembly");
            let (_, _, cache) = super::solve_arrow_newton_step_with_options(
                &system,
                1.0e-6,
                1.0e-6,
                &super::ArrowSolveOptions::direct(),
            )
            .expect("Newton metric");
            let mut sites = Vec::new();
            for row in 0..term.n_obs() {
                for (local, var) in term
                    .row_vars_for_cache_row(row, &cache)
                    .expect("cache row layout")
                    .iter()
                    .enumerate()
                {
                    if let super::SaeLocalRowVar::Logit { atom } = *var {
                        sites.push((row * term.k_atoms() + atom, cache.row_offsets[row] + local));
                    }
                }
            }
            let dim = cache.delta_t_len() + cache.k;
            let weight = ndarray::Array2::from_shape_fn((dim, dim), |(i, j)| {
                ((i + j + 1) as f64 * 0.17).cos() + if i == j { 2.0 } else { 0.0 }
            });
            let data = crate::assignment::ordered_beta_bernoulli_logit_adjoint_data_weighted(
                &term.assignment,
                &rho,
                term.row_loss_weights.as_deref(),
            )
            .expect("prior data")
            .expect("OBB family");
            let analytic = term
                .dense_exact_a_ordered_bb_logit_theta_adjoint(&cache, &weight, &data)
                .expect("full prior adjoint");
            let channels = super::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                &term.assignment,
                &rho,
                term.row_loss_weights.as_deref(),
            )
            .expect("prior channels")
            .expect("OBB family");
            assert!(channels.diagonal_term.iter().any(|&x| x > 0.0));
            assert!(channels.diagonal_term.iter().any(|&x| x < 0.0));
            let contraction = |moved: &super::SaeManifoldTerm| {
                let channels = super::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                    &moved.assignment,
                    &rho,
                    moved.row_loss_weights.as_deref(),
                )
                .expect("perturbed channels")
                .expect("OBB family");
                let mut unit = Array1::zeros(moved.assignment.logits.len());
                let mut trace = 0.0;
                for &(flat_col, global_col) in &sites {
                    unit[flat_col] = 1.0;
                    let mut column = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                        &moved.assignment, &rho, moved.row_loss_weights.as_deref(), unit.view(),
                    ).expect("existing exact HVP remainder");
                    unit[flat_col] = 0.0;
                    column[flat_col] += channels.diagonal_term[flat_col].max(0.0);
                    for &(flat_row, global_row) in &sites {
                        trace += weight[[global_row, global_col]] * column[flat_row];
                    }
                }
                trace
            };
            let h = 1.0e-5;
            let mut plus = term.clone();
            let mut minus = term.clone();
            let mut expected = 0.0;
            for &(flat, global) in &sites {
                let direction = ((flat + 1) as f64 * 0.31).sin();
                let row = flat / term.k_atoms();
                let atom = flat % term.k_atoms();
                plus.assignment.logits[[row, atom]] += h * direction;
                minus.assignment.logits[[row, atom]] -= h * direction;
                expected += analytic[global] * direction;
            }
            let fd = (contraction(&plus) - contraction(&minus)) / (2.0 * h);
            assert!(
                (fd - expected).abs() <= 1.0e-7 * (1.0 + expected.abs()),
                "weighted={weighted}: analytic={expected:e}, fd={fd:e}"
            );
        }
    }

    /// #2822 — with a learnable concentration `ρ_sparse = log(α/α_base)`, the sparse
    /// ½log|A| trace contracts the concentration derivative of the exact prior Hessian.
    /// The reference reads the exact Hessian's columns off the existing HVP remainder
    /// plus its majorizer diagonal, differentiates them centrally in `ρ_sparse` at a
    /// fixed state, and contracts against an arbitrary symmetric matrix: the trace is
    /// linear in `A⁺`, so this isolates the channel algebra from the pseudo-inverse.
    #[test]
    fn ordered_bb_learnable_alpha_sparse_trace_matches_exact_prior_hessian_difference_2822() {
        for weighted in [false, true] {
            let (mut term, target, rho) =
                crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.01, -1.0);
            let super::AssignmentMode::OrderedBetaBernoulli {
                temperature, alpha, ..
            } = term.assignment.mode
            else {
                panic!("the patch-D fixture is an ordered Beta--Bernoulli term");
            };
            term.assignment.mode =
                super::AssignmentMode::ordered_beta_bernoulli(temperature, alpha, true);
            assert!(term.assignment.effective_alpha_is_learnable());
            if weighted {
                term.set_row_loss_weights(
                    (0..term.n_obs())
                        .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                        .collect(),
                )
                .expect("positive design weights");
            }
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("learnable-concentration OBB assembly");
            let (_, _, cache) = super::solve_arrow_newton_step_with_options(
                &system,
                1.0e-6,
                1.0e-6,
                &super::ArrowSolveOptions::direct(),
            )
            .expect("Newton metric");
            let mut sites = Vec::new();
            for row in 0..term.n_obs() {
                for (local, var) in term
                    .row_vars_for_cache_row(row, &cache)
                    .expect("cache row layout")
                    .iter()
                    .enumerate()
                {
                    if let super::SaeLocalRowVar::Logit { atom } = *var {
                        sites.push((row * term.k_atoms() + atom, cache.row_offsets[row] + local));
                    }
                }
            }
            let dim = cache.delta_t_len() + cache.k;
            let weight = ndarray::Array2::from_shape_fn((dim, dim), |(i, j)| {
                ((i + j + 1) as f64 * 0.17).cos() + if i == j { 2.0 } else { 0.0 }
            });
            let analytic = term
                .dense_exact_a_ordered_bb_sparse_trace(&rho, &cache, &weight)
                .expect("learnable-concentration sparse trace");
            let contraction = |moved: &super::SaeManifoldRho| {
                let channels = super::ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
                    &term.assignment,
                    moved,
                    term.row_loss_weights.as_deref(),
                )
                .expect("perturbed channels")
                .expect("OBB family");
                let mut unit = Array1::zeros(term.assignment.logits.len());
                let mut trace = 0.0;
                for &(flat_col, global_col) in &sites {
                    unit[flat_col] = 1.0;
                    let mut column = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                        &term.assignment, moved, term.row_loss_weights.as_deref(), unit.view(),
                    ).expect("existing exact HVP remainder");
                    unit[flat_col] = 0.0;
                    column[flat_col] += channels.diagonal_term[flat_col].max(0.0);
                    for &(flat_row, global_row) in &sites {
                        trace += weight[[global_row, global_col]] * column[flat_row];
                    }
                }
                0.5 * trace
            };
            let h = 1.0e-5;
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus.log_lambda_sparse += h;
            minus.log_lambda_sparse -= h;
            let fd = (contraction(&plus) - contraction(&minus)) / (2.0 * h);
            assert!(
                analytic.is_finite() && analytic != 0.0,
                "weighted={weighted}: the concentration channel must carry signal; analytic={analytic:e}"
            );
            assert!(
                (fd - analytic).abs() <= 1.0e-7 * (1.0 + analytic.abs()),
                "weighted={weighted}: analytic={analytic:e}, fd={fd:e}"
            );
        }
    }

    fn price_fixture(
        a: &ndarray::Array2<f64>,
        e: &Array1<f64>,
    ) -> Result<PricedFixture, super::SaeCriterionError> {
        let block = spectral_fixture(a);
        let basin = super::SaeManifoldTerm::classify_exact_hessian_basin(
            &block,
            e,
            None,
            e.len(),
            "fixture",
            None,
        )?;
        let differential = super::SaeManifoldTerm::exact_hessian_basin_differential(
            &block,
            e,
            None,
            e.len(),
            &basin,
        )?;
        Ok(PricedFixture {
            log_det: basin.log_det,
            a_derivative: differential.a_derivative,
            clamp_diagonal_derivative: differential.clamp_diagonal_derivative,
        })
    }

    #[test]
    fn priced_basin_is_continuous_and_basis_invariant_at_a_repeated_negative_eigenvalue_2820() {
        let e = Array1::from_vec(vec![2.0, 4.0]);
        for epsilon in [0.0, 1.0e-8, 1.0e-4] {
            let a = ndarray::arr2(&[[-1.0, epsilon], [epsilon, -1.0]]);
            let priced = price_fixture(&a, &e).expect("positive basin");
            let expected = (3.0 - epsilon * epsilon).ln();
            assert!(
                (priced.log_det - expected).abs() <= 1.0e-12,
                "epsilon={epsilon:e}: price={}, expected={expected}",
                priced.log_det
            );
        }
        let a = -ndarray::Array2::<f64>::eye(2);
        for angle in [0.0_f64, 0.31, std::f64::consts::FRAC_PI_4] {
            let mut block = spectral_fixture(&a);
            let rotation =
                ndarray::arr2(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
            block.eigenvectors = block.eigenvectors.dot(&rotation);
            let basin = super::SaeManifoldTerm::classify_exact_hessian_basin(
                &block,
                &e,
                None,
                2,
                "rotated fixture",
                None,
            )
            .expect("same negative subspace");
            let priced = super::SaeManifoldTerm::exact_hessian_basin_differential(
                &block, &e, None, 2, &basin,
            )
            .expect("same negative-subspace differential");
            assert!((basin.log_det - 3.0_f64.ln()).abs() <= 1.0e-12);
            let inverse = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0 / 3.0]]);
            assert!(
                (&priced.a_derivative - &inverse)
                    .iter()
                    .all(|x| x.abs() <= 1.0e-12)
            );
        }
    }

    #[test]
    fn priced_basin_refuses_an_indefinite_block_with_positive_diagonal_prices_2820() {
        let a = ndarray::arr2(&[[-1.0, 0.01], [0.01, -1.0]]);
        let e = Array1::from_vec(vec![0.5, 3.0]);
        let block = spectral_fixture(&a);
        for i in 0..2 {
            let vector = block.eigenvectors.column(i);
            let diagonal_price =
                block.eigenvalues[i] + (0..2).map(|r| e[r] * vector[r] * vector[r]).sum::<f64>();
            assert!(
                diagonal_price > 0.7,
                "the old diagonal-only rule must admit this fixture"
            );
        }
        assert!(matches!(
            price_fixture(&a, &e),
            Err(super::SaeCriterionError::IndefiniteObservedInformation { .. })
        ));
    }

    #[test]
    fn priced_basin_differential_covers_a_rotated_negative_cluster_and_positive_complement_2820() {
        // Eigenvalues (-1,-1,2); the positive eigenvector has equal components.
        // Nonuniform E couples the repeated negative cluster to its complement.
        let a = ndarray::arr2(&[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]);
        let e = Array1::from_vec(vec![2.0, 4.0, 3.0]);
        let priced = price_fixture(&a, &e).expect("positive restricted basin");
        let positive = Array1::from_elem(3, 1.0 / 3.0_f64.sqrt());
        let response = priced.a_derivative.dot(&positive) - positive.mapv(|x| 0.5 * x);
        assert!(
            response.dot(&response).sqrt() > 1.0e-3,
            "the projector derivative must contribute beyond the block inverse"
        );
        let h = 1.0e-5;
        for i in 0..3 {
            for j in i..3 {
                let mut plus = a.clone();
                let mut minus = a.clone();
                plus[[i, j]] += h;
                minus[[i, j]] -= h;
                if i != j {
                    plus[[j, i]] += h;
                    minus[[j, i]] -= h;
                }
                let fd = (price_fixture(&plus, &e)
                    .expect("positive A endpoint")
                    .log_det
                    - price_fixture(&minus, &e)
                        .expect("negative A endpoint")
                        .log_det)
                    / (2.0 * h);
                let analytic = priced.a_derivative[[i, j]] * if i == j { 1.0 } else { 2.0 };
                assert!(
                    (fd - analytic).abs() <= 1.0e-7,
                    "A[{i},{j}]: analytic={analytic:e}, fd={fd:e}"
                );
            }
            let mut plus = e.clone();
            let mut minus = e.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (price_fixture(&a, &plus)
                .expect("positive E endpoint")
                .log_det
                - price_fixture(&a, &minus)
                    .expect("negative E endpoint")
                    .log_det)
                / (2.0 * h);
            assert!(
                (fd - priced.clamp_diagonal_derivative[i]).abs() <= 1.0e-7,
                "E[{i}]: analytic={}, fd={fd:e}",
                priced.clamp_diagonal_derivative[i]
            );
        }
    }

    #[test]
    fn priced_clamp_explicit_theta_derivative_has_full_logdet_scale_2820() {
        let (mut term, target, rho) =
            crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture(
                true,
            );
        let (_, _, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("fixed-state clamp fixture");
        let geometry = term
            .materialize_dense_exact_a_geometry(&rho, target.view(), &cache)
            .expect("priced spectral geometry");
        let pricing = super::SaeManifoldTerm::price_exact_hessian_block(
            &geometry.block,
            &geometry.e_diag,
            geometry.e_beta.as_ref(),
            geometry.total_t,
            "joint",
        )
        .expect("priced spectral pricing");
        let h = 1.0e-6;
        let mut live = 0;
        {
            let (block, pricing) = (&geometry.block, &pricing);
            let (_, analytic) = term
                .priced_clamp_adjoint_extras(&rho, &cache, pricing)
                .expect("explicit clamp derivative");
            // Hold A's eigensystem fixed to isolate dE. The companion K
            // contraction owns its eigenvector response, tested end to end by
            // the sparse logdet trace gate. The reference is the classifier's
            // own basin log-determinant at the moved E, so it prices exactly the
            // modes the criterion prices: log(mu) above the direction floor and
            // nothing at or inside it, where Gamma carries no weight either. It
            // prices log(mu) at full scale, so an accidental leading half in
            // Gamma cannot pass.
            assert!(
                (0..block.eigenvalues.len())
                    .any(|index| block.eigenvalues[index] < -block.rank_floor(index)),
                "fixture must price a negative direction"
            );
            let value = |moved: &super::SaeManifoldTerm| {
                let e = moved
                    .materialize_ard_concave_clamp_diagonal(&rho, &cache)
                    .expect("perturbed clamp");
                // #2828 gave E a border block, the decoder priors' majorization
                // gap, and the priced basin carries it, so this reference does.
                let e_beta = moved
                    .decoder_prior_majorizer_gap_border(&cache)
                    .expect("perturbed border gap");
                super::SaeManifoldTerm::classify_exact_hessian_basin(
                    block,
                    &e,
                    e_beta.as_ref(),
                    e.len(),
                    "joint",
                    None,
                )
                .expect("reference basin")
                .log_det
            };
            for row in 0..term.n_obs() {
                for (local, variable) in term
                    .row_vars_for_cache_row(row, &cache)
                    .expect("row variables")
                    .iter()
                    .enumerate()
                {
                    // A clone drops the three frozen gates, and
                    // `barrier_coactivation_pairs` then recomputes the barrier
                    // coactivation from the moved logits, so the border gap would
                    // move with theta. Production holds the gates fixed across a step.
                    let mut plus = term.clone();
                    let mut minus = term.clone();
                    for endpoint in [&mut plus, &mut minus] {
                        endpoint.decoder_repulsion_gate = term.decoder_repulsion_gate.clone();
                        endpoint.barrier_coactivation_gate = term.barrier_coactivation_gate.clone();
                        endpoint.amplitude_barrier_gate = term.amplitude_barrier_gate;
                        endpoint.streaming_gates_frozen = true;
                    }
                    match *variable {
                        super::SaeLocalRowVar::Logit { atom } => {
                            plus.assignment.logits[[row, atom]] += h;
                            minus.assignment.logits[[row, atom]] -= h;
                        }
                        super::SaeLocalRowVar::Coord { atom, axis } => {
                            let index = row * term.assignment.coords[atom].latent_dim() + axis;
                            let mut p = plus.assignment.coords[atom].as_flat().clone();
                            let mut m = minus.assignment.coords[atom].as_flat().clone();
                            p[index] += h;
                            m[index] -= h;
                            plus.assignment.coords[atom].set_flat(p.view());
                            minus.assignment.coords[atom].set_flat(m.view());
                        }
                    }
                    let fd = (value(&plus) - value(&minus)) / (2.0 * h);
                    let expected = analytic[cache.row_offsets[row] + local];
                    assert!(
                        (fd - expected).abs() <= 1.0e-6 + 1.0e-5 * expected.abs(),
                        "row={row}, variable={variable:?}: analytic={expected:e}, fd={fd:e}"
                    );
                    live += usize::from(expected.abs() > 1.0e-3);
                }
            }
        }
        assert!(live > 0, "the explicit theta remainder must carry signal");
    }

    impl super::SaeManifoldTerm {
        /// The dense joint arrow inverse `G = H⁻¹` (`dim×dim`), materialized
        /// column by column against each unit arrow basis vector and symmetrized:
        /// the dense reference the θ-adjoint parity tests contract against.
        /// `solver` must be `DeflatedArrowSolver::plain`.
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
            let floor = super::sae_exact_a_pencil_floor() * spectral_norm;
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
            let geometry = self.materialize_dense_exact_a_geometry(rho, target, cache)?;
            let pricing = Self::price_exact_hessian_block(
                &geometry.block,
                &geometry.e_diag,
                geometry.e_beta.as_ref(),
                geometry.total_t,
                "joint",
            )
            .map_err(|error| error.to_string())?;
            let (_, clamp_gamma) = self.priced_clamp_adjoint_extras(rho, cache, &pricing)?;
            let mut gamma = self.logdet_theta_adjoint_dense(
                rho,
                cache,
                &pricing.a_derivative,
                true,
                true,
                Some(target),
            )?;
            gamma.t += &clamp_gamma;
            gamma.beta += &self.decoder_prior_gap_theta_trace(
                cache, pricing.clamp_border_derivative.view(),
            )?;
            let (_, metric_gamma) = self.evidence_metric_derivative_channels(
                rho,
                target,
                cache,
                &pricing.metric_derivative,
            )?;
            gamma.t += &metric_gamma.t;
            gamma.beta += &metric_gamma.beta;
            Ok(gamma)
        }
    }

    /// #2915 — channel (3b), the ThresholdGate concave remainder on logit slots,
    /// through both readers of `ΔC`.
    ///
    /// The #2509 pin below runs a softmax fixture and never reaches (3b). The
    /// applier carried (3b) from #2520 on while the assembler did not, so the arrow
    /// exact-A system priced the majorizer `B` on every switched-on logit: in job
    /// 580711 the central difference of the assembled row block in
    /// `log_lambda_sparse` read `0` on each switched-on logit, where the remainder
    /// is `−2.74e-2` or `−3.19e-2`. The straddling fixture switches a logit on in
    /// every row, so the remainder is live on every row. The positive control
    /// removes (3b) from the assembled contraction and requires the comparison to
    /// see it.
    #[test]
    fn assembled_exact_hessian_delta_carries_the_threshold_gate_remainder_2915() {
        let (mut term, target, rho) =
            crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture(
                true,
            );
        let system = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("#2915 threshold-gate arrow assembly");
        let cache = gam_solve::arrow_schur::solve_arrow_newton_step_with_options(
            &system,
            1.0e-6,
            1.0e-6,
            &gam_solve::arrow_schur::ArrowSolveOptions::direct(),
        )
        .expect("#2915 positive Newton metric")
        .2;
        let remainder = crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
            &term.assignment,
            &rho,
            term.row_loss_weights.as_deref(),
        )
        .expect("#2915 threshold-gate remainder");
        let n = term.n_obs();
        let k_atoms = term.k_atoms();
        let live_rows = (0..n)
            .filter(|&row| (0..k_atoms).any(|atom| remainder[row * k_atoms + atom] < 0.0))
            .count();
        assert_eq!(
            live_rows, n,
            "#2915 premise: the straddling gate must switch a logit on in every row"
        );
        let blocks = term
            .assemble_exact_hessian_minus_b_rows(&rho, target.view(), &cache.row_dims, cache.k)
            .expect("#2915 assembled delta rows");
        assert_eq!(blocks.len(), n);
        let row_vars: Vec<Vec<super::SaeLocalRowVar>> = (0..n)
            .map(|row| {
                term.row_vars_for_cache_row(row, &cache)
                    .expect("#2915 row variables")
            })
            .collect();
        let total_t = cache.delta_t_len();
        let mut control_separated = false;
        for probe in 0..=total_t {
            let mut v = SaeArrowVector {
                t: Array1::<f64>::zeros(total_t),
                beta: Array1::<f64>::zeros(cache.k),
            };
            if probe < total_t {
                v.t[probe] = 1.0;
            } else {
                for (idx, value) in v.t.iter_mut().enumerate() {
                    *value = 1.0 + 0.25 * (idx as f64);
                }
            }
            let applied = term
                .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &v)
                .expect("#2915 matrix-free delta apply");
            // `v.beta = 0`, so the t components see only the assembled t–t blocks.
            let scale = applied
                .t
                .iter()
                .fold(1.0_f64, |acc, value| acc.max(value.abs()));
            let tolerance = 4096.0 * f64::EPSILON * scale;
            for (row, block) in blocks.iter().enumerate() {
                let base = cache.row_offsets[row];
                let q = cache.row_dims[row];
                assert_eq!(row_vars[row].len(), q);
                for a in 0..q {
                    let mut assembled = 0.0_f64;
                    for b in 0..q {
                        assembled += block.tt[[a, b]] * v.t[base + b];
                    }
                    assert!(
                        (applied.t[base + a] - assembled).abs() <= tolerance,
                        "probe {probe}: assembled ΔC t[{}] = {assembled} but the applier says {} \
                         (tolerance {tolerance:.3e})",
                        base + a,
                        applied.t[base + a]
                    );
                    if let super::SaeLocalRowVar::Logit { atom } = row_vars[row][a] {
                        let without_gate =
                            assembled - remainder[row * k_atoms + atom] * v.t[base + a];
                        control_separated |=
                            (applied.t[base + a] - without_gate).abs() > tolerance;
                    }
                }
            }
        }
        assert!(
            control_separated,
            "#2915 positive control: removing (3b) from the assembled contraction must break \
             the agreement, else this comparison cannot see the channel"
        );
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
    ///
    /// The applier also adds leg (5), the decoder priors' exact-minus-majorizer
    /// border block (#2828), which no row block holds. The assembled side contracts
    /// the operator the exact-A arrow system carries for it, so the gate covers
    /// both halves of the `ΔC` that system prices.
    #[test]
    fn assembled_exact_hessian_delta_contracts_like_the_applier_2509() {
        use ndarray::Array1;
        // #2681 — the assembled/applied parity below is a statement about two
        // readers of ONE derivation, evaluated at whatever `(t, β)` and cache
        // they are handed; it has no stake in the inner solve reaching a KKT
        // point. This fixture's inner solve does not reach it (#2681), so
        // demanding convergence here only prevented the parity from ever being
        // checked. Take the pinned shared state and factor once at it through
        // the production `FROZEN_INNER_STATE` freeze lane: the criterion's own
        // refresh and freeze-lane converge, without the criterion's ½log|A|
        // pricing. The pinned state is not a KKT point and its exact A is
        // indefinite, so that pricing refuses (census job 532879 read
        // `IndefiniteObservedInformation { block: "joint" }` here). That verdict is
        // about the Laplace normaliser, not about the parity pinned below.
        let (term0, target, rho) =
            crate::manifold::tests::small_two_atom_periodic_term_at_shared_inner_state();
        let mut term = term0;
        let mut rho_fixed = rho.clone();
        let frozen_refresh = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                &mut rho_fixed,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.25,
                1.0e-4,
                1.0e-4,
            )
            .expect("freeze-lane refresh at the pinned #2509 witness state");
        let mut frozen_loss = frozen_refresh.loss;
        let mut frozen_fixed_point = frozen_refresh.fixed_point;
        let options = gam_solve::arrow_schur::ArrowSolveOptions::direct()
            .with_gpu_policy(term.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.25,
                1.0e-4,
                1.0e-4,
                &mut frozen_loss,
                &mut frozen_fixed_point,
                &options,
                true,
            )
            .expect("freeze-lane factorization at the pinned #2509 witness state");

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
        let border_remainder = term
            .decoder_prior_border_remainder_op(cache.k, 1.0)
            .expect("the pinned state's border is the full-B or the factored layout");

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
                        assembled.beta[channel.index] += block.tbeta[[a, beta_pos]] * v.t[base + a];
                    }
                    assembled.t[base + a] += acc;
                }
            }
            // Leg (5), `ΔC_ββ`, is a border object the rows do not hold: the
            // exact-A arrow system carries it as the decoder-prior remainder
            // operator (#2828), so the assembled side contracts that operator.
            if let Some(remainder) = border_remainder.as_ref() {
                gam_solve::arrow_schur::BetaPenaltyOp::matvec(
                    remainder,
                    v.beta.as_slice().expect("an owned probe is contiguous"),
                    assembled
                        .beta
                        .as_slice_mut()
                        .expect("an owned accumulator is contiguous"),
                );
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

    /// A gapless near-null cluster must be removed without disturbing any
    /// resolved coordinate. The former inverse-power loop exhausted its work
    /// bound on this fixture; a spectral action classifies the cluster together.
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
        let rhs = SaeArrowVector {
            t: Array1::from_elem(dim, 1.0),
            beta: Array1::zeros(0),
        };

        let solved =
            solve_exact_stationarity_krylov(&rhs, &apply_a, &apply_b, &apply_b)
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
        // `1/NEAR_NULL = 1e10`. The pseudoinverse must instead return zero
        // in these directions, to the same accuracy as its resolved entries.
        for slot in RESOLVED..dim {
            assert!(
                solved.t[slot].abs() < 1.0e-6,
                "near-null slot {slot} still carries a 1/μ amplification: {:.3e}",
                solved.t[slot],
            );
        }
    }
}

#[cfg(test)]
mod tests_route_forced_classification_2673 {
    use super::super::tests::{TestPeriodicEvaluator, periodic_basis};
    use super::*;
    use gam_solve::arrow_schur::{ArrowSolveOptions, solve_arrow_newton_step_with_options};
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    /// #2828 item 2 — the matrix-free exact-`A` apply IS the dense one, and the
    /// two exact-stationarity solves agree on a right-hand side aimed straight
    /// at the classification band.
    ///
    /// `route_forced_stationarity_classification_agrees_2673` below reports the
    /// same comparison but does not assert it, and says why: its fixture "sits
    /// `2.8e7` bands away from any classification boundary, so it exercises the
    /// routes, not the predicate". #2828 item 2 is about a state that is IN the
    /// band, so this gate anchors on the #2330 Patch-D fixture's converged mode
    /// with its gates at the deflating temperature `τ = f^{-1/2}`, whose pencil
    /// carries 19 in-band directions (job 1265484). The deeper rung `τ = 1/f`,
    /// tried only when that one has none, is the less informative state: it puts
    /// every gate direction `f²` deep in the band, a null to rounding, where two
    /// solves compare 0/0 relative to themselves (job 1266736). The band
    /// membership is ASSERTED, so the gate cannot quietly become the
    /// far-from-the-boundary one it replaces.
    ///
    /// What it caught: the matrix-free route reaches its majorizer through
    /// `matrix_free_arrow_operator_apply`, which applies the CONDITIONED row
    /// factor, so it was building `Φ(B_raw) + ΔC` rather than `B_raw + ΔC`. On
    /// the pinned columns the two operators differed by 1.0 in absolute terms,
    /// and on a right-hand side aligned with the smallest eigendirection the
    /// matrix-free "solution" had `‖Ax − rhs‖ = ‖rhs‖` — no residual reduction —
    /// while the `μ` deflation detector read 0.99 and accepted it, because it
    /// inspects the solution and that solution had no near-null component to
    /// detect. See [`SaeManifoldTerm::apply_exact_hessian_matrix_free`].
    #[test]
    fn matrix_free_exact_a_matches_the_dense_operator_and_solve_in_the_band_2828() {
        use crate::manifold::tests_deflated_from_probes_2712::deflating_gate_temperatures;
        use crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture;
        // #2933 F07 — the band is a property of the pencil `(A, Φ)`, read off production's
        // own geometry. The gate is only about the classification band, so it aims at the
        // in-band direction nearest its own edge; an empty band would leave the two routes
        // nothing to classify differently, which is why #2673's report is not evidence about
        // the predicate.
        //
        // The in-band directions are saturated gate directions, which production's evidence
        // factor pins. At the fixture's historical gate temperature 0.7 the gate-logit Jacobian
        // (19ce8785f3) gives the gates an interior mode, so no row deflates and the band is
        // empty (min |μ|/floor 1.3e7, job 1265484). Every logit-slot curvature carries τ⁻², so
        // the gates go on the deflating temperature ladder the #2080 anchors use (ee6d82a554),
        // job 1265484 found 19 in-band directions at τ = f^{-1/2} and 20 at τ = 1/f. This gate
        // aims at the band's EDGE, so the rung that puts a logit eigenvalue at the band's own
        // scale, f^{-1/2}, comes first. At 1/f every gate direction sits f² deep, a null to
        // rounding (μ = -9.8e-17 against the floor 1.5e-8, job 1266736), where the solves'
        // null-only comparison has no scale.
        let (term, target, rho, system, cache, geometry) = deflating_gate_temperatures()
            .into_iter()
            .rev()
            .find_map(|temperature| {
                let (mut term, target, rho) = obb_patchd_fixture(0.0, -6.0);
                term.assignment.mode =
                    crate::assignment::AssignmentMode::ordered_beta_bernoulli(temperature, 0.9, false);
                if let Err(error) = term.penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    &rho,
                    None,
                    200,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                ) {
                    eprintln!("#2828 item 2: tau={temperature:.1e} has no converged mode: {error}");
                    return None;
                }
                // Reassemble the undamped system at the converged state and factor it,
                // exactly as the matrix-free route's own caller does.
                let system = term
                    .assemble_arrow_schur(target.view(), &rho, None)
                    .expect("undamped arrow-Schur assembly at the converged mode");
                let (_delta_t, _delta_beta, cache) = solve_arrow_newton_step_with_options(
                    &system,
                    0.0,
                    0.0,
                    &term.evidence_factor_options(),
                )
                .expect("undamped factor cache");
                let geometry = term
                    .materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
                    .expect("dense pencil geometry at the converged mode");
                eprintln!(
                    "#2828 item 2: tau={temperature:.1e}: {} in-band pencil directions",
                    geometry.band.len()
                );
                (!geometry.band.is_empty()).then_some((term, target, rho, system, cache, geometry))
            })
            .expect(
                "#2828 item 2: this gate is stated ON the classification band, but no deflating \
                 gate temperature gives the Patch-D fixture a converged mode with an in-band \
                 pencil direction",
            );
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let dense = term
            .materialize_exact_hessian_dense(&rho, target.view(), &cache)
            .expect("dense exact A at the converged mode");
        let spectral_norm = dense.iter().map(|value| value * value).sum::<f64>().sqrt();
        let flattest = geometry
            .band
            .iter()
            .copied()
            .max_by(|&a, &b| {
                (geometry.eigenvalues[a].abs() / geometry.rank_floor(a))
                    .total_cmp(&(geometry.eigenvalues[b].abs() / geometry.rank_floor(b)))
            })
            .expect("the accepted mode has an in-band pencil direction");
        let steepest = (0..dim)
            .max_by(|&a, &b| {
                geometry.eigenvalues[a]
                    .abs()
                    .total_cmp(&geometry.eigenvalues[b].abs())
            })
            .expect("non-empty spectrum");
        eprintln!(
            "#2828 item 2: {} in-band pencil directions; nearest its edge μ={:.6e} against \
             {:.6e}",
            geometry.band.len(),
            geometry.eigenvalues[flattest],
            geometry.rank_floor(flattest),
        );
        let metric = ArrowMetric::Joint(&cache)
            .prepare()
            .expect("prepared evidence metric");
        let smallest = metric
            .apply(geometry.eigenvectors.column(flattest))
            .expect("metric image of the flattest direction");
        let resolved = metric
            .apply(geometry.eigenvectors.column(steepest))
            .expect("metric image of the steepest direction");

        // (1) the OPERATORS, column by column.
        let mut worst_column = 0.0_f64;
        let mut worst_at = 0usize;
        for column in 0..dim {
            let mut unit = Array1::<f64>::zeros(dim);
            unit[column] = 1.0;
            let probe = SaeArrowVector {
                t: unit.slice(s![..total_t]).to_owned(),
                beta: unit.slice(s![total_t..]).to_owned(),
            };
            let applied = term
                .apply_exact_hessian_matrix_free(&rho, target.view(), &cache, &system, &probe)
                .expect("matrix-free exact-A apply");
            for row in 0..dim {
                let value = if row < total_t {
                    applied.t[row]
                } else {
                    applied.beta[row - total_t]
                };
                let error = (value - dense[[row, column]]).abs();
                if error > worst_column {
                    worst_column = error;
                    worst_at = column;
                }
            }
        }
        assert!(
            worst_column <= 1.0e-10 * spectral_norm,
            "#2828 item 2: the matrix-free exact-A apply is not the dense operator. Worst \
             column error {worst_column:.6e} at column {worst_at} against a spectral norm of \
             {spectral_norm:.6e}. Adding `ΔC` to the CONDITIONED `Φ(B_raw)` pins every \
             spectrally deflated direction to unit curvature, so the two routes' `A⁻¹` \
             responses differ by the whole `1/λ` of that direction."
        );

        // (2) the SOLVES, on a right-hand side aimed at the band: the dual images `Φw` of
        // the flattest and the steepest pencil directions. A rhs that missed the near-null
        // directions would leave the two routes nothing to classify differently — which is
        // exactly why #2673's report is not evidence about the predicate.
        for (label, flat) in [
            ("null-only", smallest.clone()),
            ("null+resolved", &smallest + &resolved),
        ] {
            let rhs = SaeArrowVector {
                t: flat.slice(s![..total_t]).to_owned(),
                beta: flat.slice(s![total_t..]).to_owned(),
            };
            let dense_solution = term
                .solve_exact_stationarity(&rho, target.view(), &cache, &rhs)
                .unwrap_or_else(|error| panic!("{label}: dense stationarity solve: {error}"));
            let free_solution = term
                .solve_exact_stationarity_matrix_free(&rho, target.view(), &cache, &system, &rhs)
                .unwrap_or_else(|error| panic!("{label}: matrix-free stationarity solve: {error}"));
            let flatten = |x: &SaeArrowVector| -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(dim);
                out.slice_mut(s![..total_t]).assign(&x.t);
                out.slice_mut(s![total_t..]).assign(&x.beta);
                out
            };
            let norm = |x: &Array1<f64>| x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let dense_flat = flatten(&dense_solution);
            let free_flat = flatten(&free_solution);
            let rhs_norm = norm(&flat);
            // The part of the right-hand side the pencil band removes, `ΦW_Z W_Zᵀ rhs`.
            let band_coefficients = Array1::from_iter(
                geometry
                    .band
                    .iter()
                    .map(|&index| geometry.eigenvectors.column(index).dot(&flat)),
            );
            let projected = &flat - &geometry.band_metric_images.dot(&band_coefficients);
            // Both must actually SOLVE. The defect this gate was written for was
            // silent precisely because the matrix-free route returned a finite
            // vector that reduced no residual at all.
            for (who, solution) in [("dense", &dense_flat), ("matrix-free", &free_flat)] {
                let residual = norm(&(&dense.dot(solution) - &projected));
                assert!(
                    residual <= 1.0e-6 * rhs_norm,
                    "#2828 item 2 ({label}): the {who} route returned a vector with \
                     ||A x − P rhs|| = {residual:.6e} against ||rhs|| = {rhs_norm:.6e}. A \
                     residual at the scale of the right-hand side is not a solution."
                );
            }
            // The routes are compared in the pencil's own metric, `‖Δx‖_Φ`, against the scale the
            // adjoint lives at, `‖rhs‖_Φ⁻¹ / μ_min` over the retained directions, as well as the
            // solutions' own. A right-hand side inside the band has pseudoinverse zero, so each
            // route returns its own rounding there: the dense solve 5.4e-10, at its own
            // Φ-orthonormality floor of 5.9e-10, and the Krylov solve 1.4e-9 (job 1276139). A
            // comparison relative to those alone is 0/0. A route that kept a band direction would
            // differ by that direction's `1/μ`, about 2e8 here, in the same norm.
            let metric_norm = |x: &Array1<f64>| -> f64 {
                x.dot(&metric.apply(x.view()).expect("metric image")).max(0.0).sqrt()
            };
            let dual_rhs_norm = norm(&geometry.eigenvectors.t().dot(&flat));
            let smallest_retained = (0..dim)
                .filter(|index| !geometry.band.contains(index))
                .map(|index| geometry.eigenvalues[index].abs())
                .fold(f64::INFINITY, f64::min);
            let scale = metric_norm(&dense_flat)
                .max(metric_norm(&free_flat))
                .max(dual_rhs_norm / smallest_retained);
            let difference = metric_norm(&(&dense_flat - &free_flat));
            eprintln!(
                "#2828 item 2 ({label}): |x_dense|_Φ = {:.6e}, |x_free|_Φ = {:.6e}, difference \
                 {difference:.6e} against the scale {scale:.6e} (|rhs|_Φ⁻¹/μ_min = {:.6e})",
                metric_norm(&dense_flat),
                metric_norm(&free_flat),
                dual_rhs_norm / smallest_retained,
            );
            assert!(
                difference <= 1.0e-6 * scale,
                "#2828 item 2 ({label}): the dense and matrix-free exact-stationarity solves \
                 disagree by {difference:.6e} against a solution scale of {scale:.6e}. The \
                 IFT adjoint `a = A⁺Γ` would then depend on which route the working-set \
                 predicate picked — i.e. on ambient free memory."
            );
        }
    }

    /// #2673 — FORCE both classification routes on ONE state and compare.
    ///
    /// ## Why this test exists, and what it now guards
    ///
    /// Two floors used to classify directions of the same `A = B + ΔC`:
    /// `SAE_EXACT_A_PD_FLOOR_REL` (`1e-9·max(λ_max(A), 1)`, absolute) on the
    /// dense spectral path, and `√ε` on the pencil curvature
    /// `μ = xᵀAx/xᵀBx` in the former inverse-power solver. They
    /// are now ONE predicate in ONE metric — see
    /// [`sae_exact_a_pencil_floor`] and
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
    ///   `solve_exact_stationarity_krylov`.
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
    /// `solve_exact_stationarity_matrix_free` (the Ritz pseudoinverse)
    /// — with the same rhs. If the two rules classify the same
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
                let dt =
                    d.t.iter()
                        .zip(m.t.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                let db = d
                    .beta
                    .iter()
                    .zip(m.beta.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                let scale =
                    d.t.iter()
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

/// The column-loop oracle for [`SaeManifoldTerm::materialize_exact_hessian_dense`]:
/// one Hessian-vector apply per column of the `dim × dim` matrix, exact for ANY
/// operator, arrow or not. Test-only: production never pays for it, and the
/// equality pin in `tests_sparse_curvature_operator_2500` measures the probe
/// assembly against it.
#[cfg(test)]
mod column_loop_oracle_tests {
    use super::*;

    impl SaeManifoldTerm {
        pub(crate) fn materialize_exact_hessian_dense_by_columns(
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
            // #2828 — one β-tier decoder-prior plan for all `dim` columns, so
            // this oracle and the probe assembly it checks pay the same per-apply
            // cost and their timings stay comparable.
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
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
                let av = self.apply_exact_hessian_prepared(
                    rho, cache, &unit, &prepared, &residual,
                )?;
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
    }
}

// The un-prepared wrappers below build a decoder-prior plan per call; production
// callers all go through the `_prepared` forms with a plan built once per state
// (#2828), so the wrappers are test-only conveniences and live here to keep the
// workspace `warnings = "deny"` gate green (dead_code otherwise).
#[cfg(test)]
mod tests_exact_hessian_apply_wrappers {
    use super::*;

    impl SaeManifoldTerm {
        pub(crate) fn apply_exact_hessian(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            v: &SaeArrowVector,
        ) -> Result<SaeArrowVector, String> {
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            self.apply_exact_hessian_prepared(rho, cache, v, &prepared, &residual)
        }
        pub(crate) fn apply_exact_hessian_minus_b(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            v: &SaeArrowVector,
        ) -> Result<SaeArrowVector, String> {
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            self.apply_exact_hessian_minus_b_prepared(rho, cache, v, &prepared, &residual)
        }
        pub(crate) fn apply_exact_hessian_matrix_free(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
            system: &ArrowSchurSystem,
            vector: &SaeArrowVector,
        ) -> Result<SaeArrowVector, String> {
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let residual = self.prepare_residual_curvature_rows(target, cache)?;
            self.apply_exact_hessian_matrix_free_prepared(
                rho, cache, system, vector, &prepared, &residual,
            )
        }
    }
}

// The residual-currency damped path the terminal polish globalized on before
// #2080 moved acceptance to the penalized objective. It stays as a TEST ORACLE:
// the #2762 pins hold it to the pseudoinverse at ν = 0 and to the exact
// closed-form model residual `g + AΔ(ν) = Σ_i u_i c_i ν/(λ_i² + ν)`, and the
// #2080 pin uses it to show the objective step descends where this one ascends.
#[cfg(test)]
mod tests_damped_residual_path {
    use super::*;

    /// One point of the damped residual path.
    pub(crate) struct DampedResidualPathPoint {
        /// `Δ(ν)`.
        pub(crate) step: SaeArrowVector,
        /// `g + AΔ(ν)`, in the same arrow layout as the residual handed in.
        pub(crate) model_residual: SaeArrowVector,
        /// `‖Δ(ν)‖²`.
        pub(crate) step_norm_sq: f64,
        /// Directions whose damped denominator cleared the null band.
        pub(crate) retained_rank: usize,
    }

    impl ExactHessianSpectralBlock {
        /// `residual` is the stationarity residual `g`; the step returned solves
        /// the damped system against `−g`, and the modeled residual is reported
        /// for `g` itself. A direction whose damped denominator `λ² + ν` is inside
        /// the null band (`≤ rank_floor²`) contributes nothing to the step and its
        /// whole coefficient to the model residual: at `ν = 0` that is exactly the
        /// pseudoinverse's own classification.
        pub(crate) fn damped_residual_step(
            &self,
            residual: &SaeArrowVector,
            nu: f64,
        ) -> Result<DampedResidualPathPoint, String> {
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
                return Err(format!(
                    "damped residual step: damping must be finite and ≥ 0; got {nu}"
                ));
            }
            let mut flat = Array1::<f64>::zeros(dim);
            flat.slice_mut(s![..total_t]).assign(&residual.t);
            flat.slice_mut(s![total_t..]).assign(&residual.beta);
            if !flat.iter().all(|value| value.is_finite()) {
                return Err("damped residual step: residual contains a non-finite value".to_string());
            }
            // The residual's dual expansion `g = Σᵢ cᵢ Φwᵢ`, `cᵢ = wᵢᵀg`. Off the band
            // `Φwᵢ = Awᵢ/μᵢ`; on it the block carries the image itself.
            let coefficients = self.eigenvectors.t().dot(&flat);
            let mut step_coefficients = Array1::<f64>::zeros(spectral_dim);
            let mut model = Array1::<f64>::zeros(dim);
            let mut retained_rank = 0usize;
            let mut band_position = 0usize;
            for index in 0..spectral_dim {
                let mu = self.eigenvalues[index];
                let floor = self.rank_floor(index);
                let null_band = floor * floor;
                let denominator = mu * mu + nu;
                let coefficient = coefficients[index];
                let image = if self.band.get(band_position) == Some(&index) {
                    band_position += 1;
                    self.band_metric_images.column(band_position - 1).to_owned()
                } else {
                    self.operator.dot(&self.eigenvectors.column(index)) / mu
                };
                let surviving = if denominator > null_band {
                    // Δ solves `(AΦ⁻¹A + νΦ) Δ = −AΦ⁻¹g` in this direction.
                    step_coefficients[index] = -mu * coefficient / denominator;
                    retained_rank += 1;
                    coefficient * nu / denominator
                } else {
                    coefficient
                };
                model.scaled_add(surviving, &image);
            }
            let solution = self.eigenvectors.dot(&step_coefficients);
            Ok(DampedResidualPathPoint {
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
    }
}

#[cfg(test)]
#[path = "tests_exact_a_probes_2828.rs"]
mod tests_exact_a_probes_2828;

#[cfg(test)]
#[path = "tests_clamp_basin_deflation_2333.rs"]
mod tests_clamp_basin_deflation_2333;

#[cfg(test)]
#[path = "tests_residual_curvature_rows_2731.rs"]
mod tests_residual_curvature_rows_2731;

#[cfg(test)]
#[path = "tests_reduced_pencil_operands_2933.rs"]
mod tests_reduced_pencil_operands_2933;

#[cfg(test)]
#[path = "tests_pencil_classification_2933.rs"]
mod tests_pencil_classification_2933;

/// #2731 — test-side names for the two halves production reads together from
/// `SaeManifoldTerm::materialize_exact_hessian_dense_with_gap_border`.
#[cfg(test)]
mod tests_dense_exact_a_names_2731 {
    use super::*;

    impl SaeManifoldTerm {
        /// The dense exact `A` alone.
        pub(crate) fn materialize_exact_hessian_dense(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<Array2<f64>, String> {
            self.materialize_exact_hessian_dense_with_gap_border(rho, target, cache)
                .map(|(a, _)| a)
        }

        /// `E_ββ` alone, from its own `k` applies of leg (5): an independent arm
        /// against the columns the dense build keeps.
        pub(crate) fn decoder_prior_majorizer_gap_border(
            &self,
            cache: &ArrowFactorCache,
        ) -> Result<Option<Array2<f64>>, String> {
            let k = cache.k;
            if k == 0 {
                return Ok(None);
            }
            let prepared = self.prepare_decoder_prior_beta_curvature(1.0);
            let projection = crate::frames::FrameProjection::new(self);
            let mut gap = Array2::<f64>::zeros((k, k));
            let mut unit = Array1::<f64>::zeros(k);
            for col in 0..k {
                unit.fill(0.0);
                unit[col] = 1.0;
                let column =
                    self.decoder_prior_gap_border_leg(cache, &prepared, &projection, unit.view())?;
                for row in 0..k {
                    gap[[row, col]] = -column[row];
                }
            }
            if gap.iter().all(|&value| value == 0.0) {
                return Ok(None);
            }
            for row in 0..k {
                for col in (row + 1)..k {
                    let average = 0.5 * (gap[[row, col]] + gap[[col, row]]);
                    gap[[row, col]] = average;
                    gap[[col, row]] = average;
                }
            }
            Ok(Some(gap))
        }
    }
}

/// #2267 — the priced `log|A|` alone, for the tests that read the value or the
/// refusal without the refused directions production descends.
#[cfg(test)]
mod tests_exact_observed_information_names_2267 {
    use super::*;

    impl SaeManifoldTerm {
        pub(crate) fn exact_observed_information_log_dets(
            &self,
            rho: &SaeManifoldRho,
            target: ArrayView2<'_, f64>,
            cache: &ArrowFactorCache,
        ) -> Result<f64, SaeCriterionError> {
            let mut saddle_directions = Vec::new();
            self.exact_observed_information_log_dets_with_saddle_directions(
                rho,
                target,
                cache,
                &mut saddle_directions,
            )
            .map(|(log_det, _)| log_det)
        }
    }
}

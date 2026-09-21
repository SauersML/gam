// #2933 F39 step 3 — the inertia-certified definite route of a dense exact-`A` pencil
// `A w = μ Φ w`: where a Sylvester count proves every pencil direction clears its band
// edge, the value, the solves and the dense inverse are read off an arrow Cholesky of `A`
// instead of the generalized eigendecomposition. Included from `construction_exact_hessian.rs`.

/// The exact stationarity Hessian as its dense materialization assembles it, before the
/// ordered Beta–Bernoulli mass carriers' dense matrix is added (#2933 F39):
///
/// ```text
///   A = arrow(rows, border) + Σ_k c_k u_k u_kᵀ
/// ```
///
/// Each row couples only to itself and the border, and each carrier couples every row's
/// logit slots through one rank-one term.
pub(crate) struct ExactAArrow {
    /// Per cache row: `(A_tt^(i), A_tβ^(i))`, shapes `q_i × q_i` and `q_i × k`.
    pub(crate) rows: Vec<(Array2<f64>, Array2<f64>)>,
    /// `A_ββ`, `k × k`.
    pub(crate) border: Array2<f64>,
    /// `(c_k, [(t index, u_ik)])`, as [`SaeManifoldTerm::ordered_mass_hessian_carriers`]
    /// returns them.
    pub(crate) carriers: Vec<(f64, Vec<(usize, f64)>)>,
}

/// Why a dense exact-`A` pencil takes the spectral route rather than the definite one
/// (#2933 F39). Each is logged with the route it forces.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DefiniteRouteRefusal {
    /// A row the evidence factor deflated substitutes stiffness into `Φ`, so a direction's
    /// band edge reads `wᵀ(Φ − B_raw)w`, which no eigenvector-free bound here covers.
    DeflatedRows(usize),
    /// A mass carrier's coefficient is not negative, so it cannot be eliminated through the
    /// border as a positive block.
    CarrierSign(f64),
    /// The computed inverse of the metric's factor does not resolve `‖L⁻¹‖_F`: the
    /// residual `‖LĜ − I‖_F`, rounding included, is not below one.
    MetricInverseUnresolved { residual: f64 },
    /// The resolution bound `2c·ω·‖Φ‖_F` is not below one, so no shift clears the
    /// classification's resolution floor.
    ResolutionUnbounded { scaled: f64 },
    /// The shifted, equilibrated arrow Cholesky met a non-positive diagonal or rejected a
    /// pivot, so no certificate holds at the shift.
    PivotRejected,
    /// The unshifted factorization of `A` failed although the certificate held; its
    /// arithmetic did not resolve what the certificate proved.
    UnshiftedFactor,
}

impl std::fmt::Display for DefiniteRouteRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeflatedRows(rows) => write!(formatter, "{rows} deflated row(s)"),
            Self::CarrierSign(value) => write!(formatter, "a mass carrier coefficient {value:e} ≥ 0"),
            Self::MetricInverseUnresolved { residual } => {
                write!(formatter, "the metric factor's inverse residual {residual:.3e} is not below 1")
            }
            Self::ResolutionUnbounded { scaled } => {
                write!(formatter, "the resolution bound 2c·ω·‖Φ‖_F = {scaled:.3e} is not below 1")
            }
            Self::PivotRejected => formatter.write_str("a pivot inside its rounding band"),
            Self::UnshiftedFactor => formatter.write_str("the unshifted factorization failed"),
        }
    }
}

/// The lower Cholesky factor of a symmetric arrow matrix
/// `[[T, B], [Bᵀ, D]]`, `T = ⊕ᵢ Tᵢ`, in its natural elimination order:
/// `Tᵢ = LᵢLᵢᵀ`, `Vᵢ = (Lᵢ⁻¹Bᵢ)ᵀ`, and `D − Σᵢ VᵢVᵢᵀ = L_b L_bᵀ`. The zero fill between rows
/// is exact, so this IS the Cholesky factor of the whole matrix.
struct ArrowCholesky {
    row_offsets: Vec<usize>,
    row_factors: Vec<Array2<f64>>,
    /// Per row, `Vᵢ`, shape `border × qᵢ`.
    cross: Vec<Array2<f64>>,
    border_factor: Array2<f64>,
}

impl ArrowCholesky {
    /// Factor the arrow matrix, `None` where a pivot is rejected. The row blocks and the
    /// border Schur complement are factored by gam-linalg's scalar Cholesky, and every
    /// inner product below is a sequential sum, so each entry's accumulation length is the
    /// count of terms it sums: at most `total_t + border` (Higham, Thm 10.3's `γ_{n+1}` with
    /// `n` the whole dimension bounds every one of them).
    fn factor(
        row_offsets: &[usize],
        rows: &[Array2<f64>],
        cross_blocks: &[Array2<f64>],
        border: &Array2<f64>,
    ) -> Option<Self> {
        let width = border.nrows();
        let mut row_factors = Vec::with_capacity(rows.len());
        let mut cross = Vec::with_capacity(rows.len());
        let mut schur = border.clone();
        for (block, coupling) in rows.iter().zip(cross_blocks.iter()) {
            let factor = gam_linalg::triangular::cholesky_factor_in_place(
                block.view(),
                gam_linalg::triangular::CholeskyGuard::FiniteStrict,
            )?;
            // `Vᵢᵀ = Lᵢ⁻¹Bᵢ`, one forward substitution per border column.
            let solved =
                gam_linalg::triangular::forward_substitution_lower_matrix(factor.view(), coupling.view());
            let transposed = solved.t().to_owned();
            for left in 0..width {
                for right in 0..=left {
                    let mut sum = 0.0_f64;
                    for position in 0..transposed.ncols() {
                        sum += transposed[[left, position]] * transposed[[right, position]];
                    }
                    schur[[left, right]] -= sum;
                }
            }
            row_factors.push(factor);
            cross.push(transposed);
        }
        for left in 0..width {
            for right in 0..left {
                schur[[right, left]] = schur[[left, right]];
            }
        }
        let border_factor = if width == 0 {
            Array2::<f64>::zeros((0, 0))
        } else {
            gam_linalg::triangular::cholesky_factor_in_place(
                schur.view(),
                gam_linalg::triangular::CholeskyGuard::FiniteStrict,
            )?
        };
        Some(Self {
            row_offsets: row_offsets.to_vec(),
            row_factors,
            cross,
            border_factor,
        })
    }

    fn total_t(&self) -> usize {
        self.row_offsets.last().copied().unwrap_or(0)
    }

    fn border(&self) -> usize {
        self.border_factor.nrows()
    }

    /// `log det` of the factored matrix, `2·Σ ln Lᵢᵢ`.
    fn log_det(&self) -> f64 {
        let diagonal_log = |factor: &Array2<f64>| {
            (0..factor.nrows())
                .map(|index| factor[[index, index]].ln())
                .sum::<f64>()
        };
        2.0 * (self.row_factors.iter().map(diagonal_log).sum::<f64>()
            + diagonal_log(&self.border_factor))
    }

    /// Solve `M x = rhs` for the factored `M`, `rhs` laid out `(t, border)`.
    fn solve(&self, rhs: ArrayView1<'_, f64>) -> Array1<f64> {
        let total_t = self.total_t();
        let width = self.border();
        // Forward: `y_i = Lᵢ⁻¹ rhs_i`, then `L_b z = rhs_b − Σ Vᵢ y_i`.
        let mut forward_t = Array1::<f64>::zeros(total_t);
        let mut border_rhs = rhs.slice(s![total_t..]).to_owned();
        for (row, factor) in self.row_factors.iter().enumerate() {
            let range = self.row_offsets[row]..self.row_offsets[row + 1];
            let solved = gam_linalg::triangular::forward_substitution_lower_vector(
                factor.view(),
                rhs.slice(s![range.clone()]),
            );
            border_rhs -= &self.cross[row].dot(&solved);
            forward_t.slice_mut(s![range]).assign(&solved);
        }
        let border = if width == 0 {
            Array1::<f64>::zeros(0)
        } else {
            let forward_b = gam_linalg::triangular::forward_substitution_lower_vector(
                self.border_factor.view(),
                border_rhs.view(),
            );
            gam_linalg::triangular::back_substitution_lower_transpose(
                self.border_factor.view(),
                forward_b.view(),
            )
        };
        // Backward: `x_i = Lᵢ⁻ᵀ(y_i − Vᵢᵀ x_b)`.
        let mut out = Array1::<f64>::zeros(total_t + width);
        for (row, factor) in self.row_factors.iter().enumerate() {
            let range = self.row_offsets[row]..self.row_offsets[row + 1];
            let shifted = &forward_t.slice(s![range.clone()]) - &self.cross[row].t().dot(&border);
            out.slice_mut(s![range]).assign(
                &gam_linalg::triangular::back_substitution_lower_transpose(
                    factor.view(),
                    shifted.view(),
                ),
            );
        }
        out.slice_mut(s![total_t..]).assign(&border);
        out
    }

    /// The leading `(total_t + primal) × (total_t + primal)` block of the factored matrix's
    /// inverse. With `Z = L_b⁻¹ V L_t⁻¹` (`border × total_t`), the inverse is
    /// `[[T⁻¹ + ZᵀZ, −ZᵀL_b⁻¹], [·, L_b⁻ᵀL_b⁻¹]]`, so the latent block is a block diagonal
    /// plus a rank-`border` Gram and nothing `dim × dim` is factored.
    fn leading_inverse(&self, primal: usize) -> Array2<f64> {
        let total_t = self.total_t();
        let width = self.border();
        let dim = total_t + primal;
        // `W = V L_t⁻¹`, one row block at a time: `Wᵢ = Vᵢ Lᵢ⁻¹ = (Lᵢ⁻ᵀ Vᵢᵀ)ᵀ`.
        let mut coupled = Array2::<f64>::zeros((width, total_t));
        let mut out = Array2::<f64>::zeros((dim, dim));
        for (row, factor) in self.row_factors.iter().enumerate() {
            let range = self.row_offsets[row]..self.row_offsets[row + 1];
            let q = factor.nrows();
            let block = gam_linalg::triangular::back_substitution_lower_transpose_matrix(
                factor.view(),
                self.cross[row].t(),
            );
            coupled.slice_mut(s![.., range.clone()]).assign(&block.t());
            let identity = Array2::<f64>::eye(q);
            let inverse = gam_linalg::triangular::cholesky_solve_matrix(factor.view(), identity.view());
            out.slice_mut(s![range.clone(), range]).assign(&inverse);
        }
        if width == 0 {
            return out;
        }
        let z = gam_linalg::triangular::forward_substitution_lower_matrix(
            self.border_factor.view(),
            coupled.view(),
        );
        let latent = z.t().dot(&z);
        out.slice_mut(s![..total_t, ..total_t]).scaled_add(1.0, &latent);
        if primal > 0 {
            // `L_b⁻ᵀ L_b⁻¹` restricted to the primal border, and `−Zᵀ L_b⁻¹` beside it.
            let identity = Array2::<f64>::eye(width);
            let border_inverse =
                gam_linalg::triangular::cholesky_solve_matrix(self.border_factor.view(), identity.view());
            let lower_inverse = gam_linalg::triangular::forward_substitution_lower_matrix(
                self.border_factor.view(),
                identity.view(),
            );
            let cross_inverse = -z.t().dot(&lower_inverse);
            out.slice_mut(s![..total_t, total_t..])
                .assign(&cross_inverse.slice(s![.., ..primal]));
            out.slice_mut(s![total_t.., ..total_t])
                .assign(&cross_inverse.slice(s![.., ..primal]).t());
            out.slice_mut(s![total_t.., total_t..])
                .assign(&border_inverse.slice(s![..primal, ..primal]));
        }
        out
    }
}

/// The certificate the definite route stands on, as numbers the log and the pins read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DefiniteRouteCertificate {
    /// The shift `σ` at which `A − σΦ ≻ 0` was proved: every pencil direction, and every
    /// computed eigenvalue within its resolution of it, clears its band edge.
    pub(crate) shift: f64,
    /// `ω ≥ max ‖w‖₂²` over `Φ`-normalized directions.
    pub(crate) metric_inverse_trace: f64,
    /// The equilibrated rounding band the shifted factorization cleared.
    pub(crate) equilibrated_band: f64,
}

/// A dense exact-`A` pencil the certificate proved definite beyond every band edge
/// (#2933 F39). Everything a spectral block with an empty band and no negative direction
/// would report is read off the factors: `½log|A|`, `½log|Φ|`, `A⁻¹v` and the dense `A⁻¹`.
pub(crate) struct DefiniteExactA {
    /// The carrier-augmented arrow Cholesky of `A` itself (no shift).
    factor: ArrowCholesky,
    /// `Σ_k ln(−c_k)`: `log|A| = log|Aug| − log|−C⁻¹|`.
    carrier_log_det: f64,
    /// `k`, the primal border; the augmented border also carries one column per carrier.
    primal_border: usize,
    pub(crate) metric_log_det: f64,
    pub(crate) certificate: DefiniteRouteCertificate,
}

impl DefiniteExactA {
    /// `log|A|`: every direction is retained and positive, so the priced log-determinant
    /// is the operator's own.
    pub(crate) fn log_det(&self) -> f64 {
        self.factor.log_det() + self.carrier_log_det
    }

    pub(crate) fn dim(&self) -> usize {
        self.factor.total_t() + self.primal_border
    }

    /// `A⁻¹v`: the augmented solve with the carrier rows' right-hand side zero, whose
    /// leading block is `(X + UCUᵀ)⁻¹v`.
    pub(crate) fn solve(&self, rhs: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let dim = self.dim();
        if rhs.len() != dim {
            return Err(format!(
                "definite exact-A solve: right-hand side of length {} for dimension {dim}",
                rhs.len()
            ));
        }
        let mut augmented = Array1::<f64>::zeros(dim + self.factor.border() - self.primal_border);
        augmented.slice_mut(s![..dim]).assign(&rhs);
        Ok(self.factor.solve(augmented.view()).slice(s![..dim]).to_owned())
    }

    /// The dense `A⁻¹`, the leading block of the augmented inverse (#2933 F39): what a
    /// spectral block's retained pseudo-inverse `W_R diag(1/μ_R) W_Rᵀ` is when every
    /// direction is retained.
    pub(crate) fn dense_inverse(&self) -> Array2<f64> {
        self.factor.leading_inverse(self.primal_border)
    }
}


/// An arrow matrix's blocks, laid out as [`ArrowCholesky::factor`] reads them.
struct ArrowBlocks {
    rows: Vec<Array2<f64>>,
    /// Per row, `Bᵢ`, shape `qᵢ × border`.
    cross: Vec<Array2<f64>>,
    border: Array2<f64>,
}

impl ArrowBlocks {
    fn diagonal(&self) -> Vec<f64> {
        self.rows
            .iter()
            .flat_map(|block| (0..block.nrows()).map(|index| block[[index, index]]).collect::<Vec<_>>())
            .chain((0..self.border.nrows()).map(|index| self.border[[index, index]]))
            .collect()
    }

    /// #2933 F39 — prove the exact matrix `M` definite from its computed blocks `M̂`, given
    /// `formation`, a bound on `‖M̂ − M‖_F`. Returns the equilibrated band the proof
    /// cleared, `None` where it does not hold.
    ///
    /// Let `d = diag(M̂) > 0`, `D = diag(d)`, and factor `M̂ − βD`.
    /// * **Factorization.** Success makes the computed factor `R̂` the exact Cholesky factor of
    ///   `M̂ − βD + E`, with `|E| ≤ γ_{n+1}|R̂||R̂ᵀ|` (Higham, *Accuracy and Stability of
    ///   Numerical Algorithms*, Thm 10.3). Row `i` of `R̂` has squared norm
    ///   `(1 − β)dᵢ + Eᵢᵢ ≤ dᵢ/(1 − γ_{n+1})`, so `|E|ᵢⱼ ≤ γ_{n+1}/(1 − γ_{n+1})·√(dᵢdⱼ)`, only
    ///   on the arrow's pattern `P`. Hence `‖D^{-½}ED^{-½}‖₂ ≤ γ_{n+1}/(1 − γ_{n+1})·‖P‖₂`, and
    ///   `‖P‖₂ ≤ max(q, b) + √(total_t·b)`: the row and border blocks, plus the all-ones
    ///   coupling between them.
    /// * **Formation.** `‖D^{-½}(M̂ − M)D^{-½}‖₂ ≤ formation / min dᵢ`.
    /// * **Conclusion.** With `β` their sum, `D^{-½}MD^{-½} ≻ βI − D^{-½}(E + M̂ − M)D^{-½} ⪰ 0`.
    fn certify_definite(&self, row_offsets: &[usize], formation: f64) -> Option<f64> {
        let diagonal = self.diagonal();
        if diagonal.iter().any(|value| !(value.is_finite() && *value > 0.0)) {
            return None;
        }
        let total_t = row_offsets.last().copied().unwrap_or(0);
        let width = self.border.nrows();
        let dim = diagonal.len();
        let max_q = self.rows.iter().map(|block| block.nrows()).max().unwrap_or(0);
        let pattern_norm = max_q.max(width) as f64 + ((total_t * width) as f64).sqrt();
        let growth = gam_linalg::roundoff::accumulation_growth(dim + 1);
        if !(growth < 1.0) {
            return None;
        }
        let smallest = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
        let band = growth / (1.0 - growth) * pattern_norm + formation / smallest;
        if !(band < 1.0) {
            return None;
        }
        let mut rows = self.rows.clone();
        let mut position = 0usize;
        for block in &mut rows {
            for index in 0..block.nrows() {
                block[[index, index]] -= band * diagonal[position];
                position += 1;
            }
        }
        let mut border = self.border.clone();
        for index in 0..width {
            border[[index, index]] -= band * diagonal[total_t + index];
        }
        ArrowCholesky::factor(row_offsets, &rows, &self.cross, &border)
            .is_some()
            .then_some(band)
    }
}

impl SaeManifoldTerm {
    /// #2933 F39 step 3 — certify the dense exact-`A` pencil `(A, Φ)` definite beyond every
    /// direction's band edge without its eigenvectors, and factor `A` for the value and the
    /// solves; or name why the spectral route must decompose it.
    ///
    /// # The band edge this must clear
    ///
    /// A direction `w` (`wᵀΦw = 1`, `Aw = μΦw`) is retained positive when
    /// `μ > sae_exact_a_band_edge(μ, τ, s)`, with both terms from gam-solve's owners
    /// (`arrow_schur/solve_options.rs`):
    ///
    /// ```text
    ///   exact_a_pencil_resolution(dim, ‖w‖₂², ‖A‖_F, ‖Φ‖_F, μ) = dim·ε·‖w‖₂²·(‖A‖_F + |μ|·‖Φ‖_F)
    ///   exact_a_band_edge(μ, τ, s)  = max(√ε, τ)       for μ ≤ 0
    ///                                = max(√ε, τ, s)    for μ > 0,     s = wᵀ(Φ − B_raw)w
    /// ```
    ///
    /// `s` is the direction's substituted stiffness (`PreparedArrowMetric::substituted_image`
    /// is `(Φ − B_raw)v` on the deflated rows' spectra and zero elsewhere). It is zero for every
    /// `w` when no row deflated; a deflated row takes the spectral route.
    ///
    /// # The shift
    ///
    /// Let `c = dim·ε`, and `ω ≥ ‖w‖₂²` for every `Φ`-normalized `w`. Since
    /// `‖w‖₂² ≤ 1/λ_min(Φ) ≤ ‖L⁻¹‖_F²` for `Φ = LLᵀ`, `ω` bounds the exact `‖L⁻¹‖_F²`. Its
    /// computed inverse `Ĝ` satisfies `L⁻¹ = Ĝ(LĜ)⁻¹`, so `‖L⁻¹‖_F ≤ ‖Ĝ‖_F / (1 − ‖LĜ − I‖_F)`,
    /// and the residual is charged its own rounding. Then, with
    ///
    /// ```text
    ///   σ = max( 2cω‖A‖_F / (1 − 2cω‖Φ‖_F),  (√ε + cω‖A‖_F) / (1 − cω‖Φ‖_F) ),    2cω‖Φ‖_F < 1,
    /// ```
    ///
    /// every exact `μ > σ` gives `μ > 2cω(‖A‖_F + μ‖Φ‖_F) ≥ 2τ` and
    /// `μ − τ ≥ μ(1 − cω‖Φ‖_F) − cω‖A‖_F > √ε`. So any eigenvalue within its own resolution `τ`
    /// of the exact one, which is what the spectral route's resolution states of its computed
    /// eigenvalues, still clears its edge `max(√ε, τ)`. Both routes retain every direction and
    /// price `log|A|`.
    ///
    /// # The Sylvester count
    ///
    /// `Φ ≻ 0`, so `A − σΦ ≻ 0` iff every `μ > σ`. `A = X + UCUᵀ` with `X` the arrow part and
    /// `C = diag(c_k) ≺ 0`, and with the carrier block `−C⁻¹ ≻ 0` appended to the border,
    /// Haynsworth gives
    ///
    /// ```text
    ///   inertia[[X − σΦ, U], [Uᵀ, −C⁻¹]] = inertia(−C⁻¹) + inertia(A − σΦ).
    /// ```
    ///
    /// The augmented matrix is an arrow with border `k + K`, and [`ArrowBlocks::certify_definite`]
    /// proves it definite from its computed blocks with every pivot outside its rounding band.
    /// Its formation charges `σΦ` formed off the metric's factor `L = [[L_t, 0], [X, L_b]]`,
    /// entrywise `γ_{total_t+k+3}·(|A| + σ|L||Lᵀ|)`, with `‖|L||Lᵀ|‖_F ≤ ‖L‖_F²`.
    pub(crate) fn certify_definite_exact_a(
        arrow: &ExactAArrow,
        metric: &PreparedArrowMetric<'_>,
    ) -> Result<Result<DefiniteExactA, DefiniteRouteRefusal>, String> {
        use gam_linalg::roundoff::accumulation_growth;
        let cache = metric.cache;
        if metric.lift.is_some() {
            return Err("certify_definite_exact_a: the definite route reads the joint metric".into());
        }
        let deflated = cache
            .deflated_row_directions
            .iter()
            .filter(|directions| !directions.is_empty())
            .count();
        if deflated > 0 {
            return Ok(Err(DefiniteRouteRefusal::DeflatedRows(deflated)));
        }
        if let Some(&(coefficient, _)) = arrow.carriers.iter().find(|(c, _)| !(*c < 0.0)) {
            return Ok(Err(DefiniteRouteRefusal::CarrierSign(coefficient)));
        }
        let n_rows = cache.n_rows();
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let carriers = arrow.carriers.len();
        let width = k + carriers;
        let dim = total_t + k;
        if arrow.rows.len() != n_rows || arrow.border.dim() != (k, k) {
            return Err(format!(
                "certify_definite_exact_a: {} arrow rows and a {:?} border for a cache of {n_rows} \
                 rows and border {k}",
                arrow.rows.len(),
                arrow.border.dim()
            ));
        }
        let row_offsets: Vec<usize> = cache.row_offsets.to_vec();

        // The metric's blocks off its factor: `Φ_tt⁽ⁱ⁾ = LᵢLᵢᵀ`, `Φ_tβ⁽ⁱ⁾ = Lᵢ Xᵢᵀ` and
        // `Φ_ββ = L_bL_bᵀ + XXᵀ`; its computed inverse `Ĝ = [[L_t⁻¹, 0], [−L_b⁻¹XL_t⁻¹, L_b⁻¹]]`
        // and the residual `LĜ − I`.
        let mut metric_rows = Vec::with_capacity(n_rows);
        let mut metric_cross = Vec::with_capacity(n_rows);
        let mut factor_sq = 0.0_f64;
        let mut inverse_sq = 0.0_f64;
        let mut residual_sq = 0.0_f64;
        let mut coupled = Array2::<f64>::zeros((k, total_t));
        for row in 0..n_rows {
            let range = row_offsets[row]..row_offsets[row + 1];
            let factor = cache.undamped_factor(row).to_owned();
            let row_coupling = metric.cross.slice(s![.., range.clone()]).to_owned();
            metric_rows.push(factor.dot(&factor.t()));
            metric_cross.push(factor.dot(&row_coupling.t()));
            factor_sq += factor.iter().map(|v| v * v).sum::<f64>()
                + row_coupling.iter().map(|v| v * v).sum::<f64>();
            let identity = Array2::<f64>::eye(factor.nrows());
            let inverse = gam_linalg::triangular::forward_substitution_lower_matrix(
                factor.view(),
                identity.view(),
            );
            inverse_sq += inverse.iter().map(|v| v * v).sum::<f64>();
            residual_sq += (&factor.dot(&inverse) - &identity).iter().map(|v| v * v).sum::<f64>();
            coupled.slice_mut(s![.., range]).assign(&row_coupling.dot(&inverse));
        }
        let metric_border = if k == 0 {
            Array2::<f64>::zeros((0, 0))
        } else {
            let lower = &metric.border_lower;
            factor_sq += lower.iter().map(|v| v * v).sum::<f64>();
            let identity = Array2::<f64>::eye(k);
            let border_inverse =
                gam_linalg::triangular::forward_substitution_lower_matrix(lower.view(), identity.view());
            let lower_coupled =
                gam_linalg::triangular::forward_substitution_lower_matrix(lower.view(), coupled.view());
            inverse_sq += border_inverse.iter().map(|v| v * v).sum::<f64>()
                + lower_coupled.iter().map(|v| v * v).sum::<f64>();
            // The border rows of `LĜ − I`: `XL_t⁻¹ − L_b(L_b⁻¹XL_t⁻¹)` and `L_bL_b⁻¹ − I`.
            residual_sq += (&coupled - &lower.dot(&lower_coupled)).iter().map(|v| v * v).sum::<f64>()
                + (&lower.dot(&border_inverse) - &identity).iter().map(|v| v * v).sum::<f64>();
            lower.dot(&lower.t()) + metric.cross.dot(&metric.cross.t())
        };
        // `|fl(LĜ) − LĜ| ≤ γ_dim·|L||Ĝ|`, so the computed residual is charged
        // `γ_dim·‖L‖_F·‖Ĝ‖_F`.
        let residual = residual_sq.sqrt()
            + accumulation_growth(dim) * factor_sq.sqrt() * inverse_sq.sqrt();
        if !(residual < 1.0) {
            return Ok(Err(DefiniteRouteRefusal::MetricInverseUnresolved { residual }));
        }
        let inflate = |value: f64, operations: usize| gam_math::roundoff::inflated(value, operations);
        let omega = inflate((inverse_sq.sqrt() / (1.0 - residual)).powi(2), dim + 3);

        // The shift.
        let arrow_frobenius_sq = arrow
            .rows
            .iter()
            .map(|(tt, tbeta)| {
                tt.iter().map(|v| v * v).sum::<f64>() + 2.0 * tbeta.iter().map(|v| v * v).sum::<f64>()
            })
            .sum::<f64>()
            + arrow.border.iter().map(|v| v * v).sum::<f64>();
        let carrier_frobenius = arrow
            .carriers
            .iter()
            .map(|(coefficient, entries)| {
                coefficient.abs() * entries.iter().map(|(_, value)| value * value).sum::<f64>()
            })
            .sum::<f64>();
        let a_frobenius = inflate(arrow_frobenius_sq.sqrt() + carrier_frobenius, dim + 3);
        let metric_frobenius = inflate(metric.frobenius_norm()?, dim + 3);
        let resolution = dim as f64 * f64::EPSILON * omega;
        let scaled = 2.0 * resolution * metric_frobenius;
        if !(scaled < 1.0) {
            return Ok(Err(DefiniteRouteRefusal::ResolutionUnbounded { scaled }));
        }
        let shift = (2.0 * resolution * a_frobenius / (1.0 - scaled)).max(
            (sae_exact_a_pencil_floor() + resolution * a_frobenius)
                / (1.0 - resolution * metric_frobenius),
        );

        // The carrier-augmented arrow of `A − σΦ`.
        let augmented = |sigma: f64| -> ArrowBlocks {
            let mut rows = Vec::with_capacity(n_rows);
            let mut cross = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let range = row_offsets[row]..row_offsets[row + 1];
                let (tt, tbeta) = &arrow.rows[row];
                rows.push(tt - &(&metric_rows[row] * sigma));
                let mut coupling = Array2::<f64>::zeros((range.len(), width));
                coupling
                    .slice_mut(s![.., ..k])
                    .assign(&(tbeta - &(&metric_cross[row] * sigma)));
                for (index, (_, entries)) in arrow.carriers.iter().enumerate() {
                    for &(position, value) in entries {
                        if range.contains(&position) {
                            coupling[[position - range.start, k + index]] = value;
                        }
                    }
                }
                cross.push(coupling);
            }
            let mut border = Array2::<f64>::zeros((width, width));
            border
                .slice_mut(s![..k, ..k])
                .assign(&(&arrow.border - &(&metric_border * sigma)));
            for (index, (coefficient, _)) in arrow.carriers.iter().enumerate() {
                border[[k + index, k + index]] = -1.0 / coefficient;
            }
            ArrowBlocks { rows, cross, border }
        };
        let carrier_inverse = arrow
            .carriers
            .iter()
            .map(|(coefficient, _)| (1.0 / coefficient).powi(2))
            .sum::<f64>()
            .sqrt();
        let formation = accumulation_growth(total_t + k + 3)
            * (a_frobenius + shift * factor_sq + carrier_inverse);
        let Some(equilibrated_band) = augmented(shift).certify_definite(&row_offsets, formation) else {
            return Ok(Err(DefiniteRouteRefusal::PivotRejected));
        };

        // `A` itself, for the value and the solves.
        let unshifted = augmented(0.0);
        let Some(factor) =
            ArrowCholesky::factor(&row_offsets, &unshifted.rows, &unshifted.cross, &unshifted.border)
        else {
            return Ok(Err(DefiniteRouteRefusal::UnshiftedFactor));
        };
        let carrier_log_det = arrow
            .carriers
            .iter()
            .map(|(coefficient, _)| (-coefficient).ln())
            .sum::<f64>();
        Ok(Ok(DefiniteExactA {
            factor,
            carrier_log_det,
            primal_border: k,
            metric_log_det: metric.log_det()?,
            certificate: DefiniteRouteCertificate {
                shift,
                metric_inverse_trace: omega,
                equilibrated_band,
            },
        }))
    }
}

#[cfg(test)]
mod definite_exact_a_tests {
    use super::*;

    const ROWS: usize = 6;
    const Q: usize = 2;
    const K: usize = 3;

    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        state ^= state >> 31;
        2.0 * ((state >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000) - 1.0
    }

    fn random_matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| scale * uniform(seed ^ ((r as u64) << 20) ^ c as u64))
    }

    fn symmetric_positive(dim: usize, seed: u64, ridge: f64) -> Array2<f64> {
        let m = random_matrix(dim, dim, seed, 1.0);
        m.dot(&m.t()) + &(Array2::<f64>::eye(dim) * ridge)
    }

    /// A definite arrow evidence system, its factor cache, and its blocks.
    fn evidence() -> (ArrowSchurSystem, ArrowFactorCache) {
        let mut system = ArrowSchurSystem::new(ROWS, Q, K);
        for row in 0..ROWS {
            system.rows[row].htt = symmetric_positive(Q, 11 + row as u64, 1.0);
            system.rows[row].htbeta = random_matrix(Q, K, 101 + row as u64, 0.3);
        }
        system.hbb = symmetric_positive(K, 7, 4.0);
        let (_, _, cache) = gam_solve::arrow_schur::solve_arrow_newton_step_with_options(
            &system,
            0.0,
            0.0,
            &ArrowSolveOptions::direct().with_positive_definite_evidence(),
        )
        .expect("the definite evidence system factors");
        (system, cache)
    }

    /// `A = scale·B + perturbation + c·u uᵀ` as an arrow plus one carrier on every row's first
    /// slot, `B` the evidence system.
    fn operator(system: &ArrowSchurSystem, scale: f64, carrier: f64) -> ExactAArrow {
        let rows = (0..ROWS)
            .map(|row| {
                let perturbation = symmetric_positive(Q, 301 + row as u64, 0.0) * 0.05;
                (
                    &system.rows[row].htt * scale + &perturbation,
                    &system.rows[row].htbeta * scale + &random_matrix(Q, K, 401 + row as u64, 0.02),
                )
            })
            .collect();
        let border = &system.hbb * scale + &(symmetric_positive(K, 501, 0.0) * 0.05);
        let carriers = if carrier == 0.0 {
            Vec::new()
        } else {
            vec![(
                carrier,
                (0..ROWS).map(|row| (row * Q, 0.5 + 0.1 * row as f64)).collect(),
            )]
        };
        ExactAArrow {
            rows,
            border,
            carriers,
        }
    }

    fn dense(arrow: &ExactAArrow) -> Array2<f64> {
        let total_t = ROWS * Q;
        let dim = total_t + K;
        let mut out = Array2::<f64>::zeros((dim, dim));
        for (row, (tt, tbeta)) in arrow.rows.iter().enumerate() {
            let base = row * Q;
            out.slice_mut(s![base..base + Q, base..base + Q]).assign(tt);
            out.slice_mut(s![base..base + Q, total_t..]).assign(tbeta);
            out.slice_mut(s![total_t.., base..base + Q]).assign(&tbeta.t());
        }
        out.slice_mut(s![total_t.., total_t..]).assign(&arrow.border);
        for (coefficient, entries) in &arrow.carriers {
            for &(left, u_left) in entries {
                for &(right, u_right) in entries {
                    out[[left, right]] += coefficient * u_left * u_right;
                }
            }
        }
        out
    }

    /// The spectral route's census: (retained, in band, negative).
    fn census(block: &ExactHessianSpectralBlock) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for index in 0..block.eigenvalues.len() {
            let mu = block.eigenvalues[index];
            if mu.abs() <= block.rank_floor(index) {
                counts.1 += 1;
            } else if mu > 0.0 {
                counts.0 += 1;
            } else {
                counts.2 += 1;
            }
        }
        counts
    }

    /// #2933 F39 — on a definite pencil the definite route reads the spectral route's value,
    /// solve and inverse off its factors.
    ///
    /// Bands. The spectral route's computed `μ̂ᵢ` sits within its resolution `τᵢ` of the exact
    /// `μᵢ`, so its `Σ ln μ̂ᵢ` sits within `Σ τᵢ/(μᵢ − τᵢ)` of `log|A| − log|Φ|`. The definite
    /// route's arrow Cholesky is backward stable with `‖ΔA‖₂ ≤ γ_{n+1}‖A‖_F·n`, which moves
    /// `log|A|` by at most `n·‖A⁻¹‖₂·‖ΔA‖₂`, and a solve by `κ₂(A)` times its relative
    /// backward error. Each comparison is also held against a magnitude floor, so a 0/0 at
    /// rounding cannot pass.
    #[test]
    fn the_definite_route_reads_the_spectral_value_solve_and_inverse_2933_f39() {
        let (system, cache) = evidence();
        let metric = ArrowMetric::Joint(&cache).prepare().expect("the evidence metric prepares");
        let arrow = operator(&system, 2.0, -0.3);
        let definite = match SaeManifoldTerm::certify_definite_exact_a(&arrow, &metric)
            .expect("the certificate evaluates")
        {
            Ok(definite) => definite,
            Err(refusal) => panic!("a definite pencil must certify, but: {refusal}"),
        };
        let dense_a = dense(&arrow);
        let dim = dense_a.nrows();
        let total_t = ROWS * Q;
        let block = SaeManifoldTerm::exact_hessian_spectral_block(dense_a.clone(), &metric)
            .expect("the spectral route decomposes the pencil");
        assert_eq!(
            census(&block),
            (dim, 0, 0),
            "premise: the spectral route retains every direction of this pencil"
        );
        assert_eq!(
            definite.metric_log_det.to_bits(),
            block.metric_log_det.to_bits(),
            "both routes read log|Φ| off the one metric factor"
        );
        eprintln!(
            "[F39 pin] shift {:.3e}, ω {:.3e}, equilibrated band {:.3e}, min μ {:.6e}",
            definite.certificate.shift,
            definite.certificate.metric_inverse_trace,
            definite.certificate.equilibrated_band,
            block.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min),
        );
        assert!(
            block.eigenvalues.iter().all(|&mu| mu > definite.certificate.shift),
            "every spectral eigenvalue must sit above the certified shift"
        );

        let (spectrum, _) = dense_a.clone().eigh(Side::Lower).expect("the dense operator decomposes");
        let a_norm = spectrum.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let a_min = spectrum.iter().copied().fold(f64::INFINITY, f64::min);
        let condition = a_norm / a_min;
        let growth = gam_linalg::roundoff::accumulation_growth(dim + K + 1);
        let a_frobenius = dense_a.iter().map(|v| v * v).sum::<f64>().sqrt();

        let spectral_log_det = SaeManifoldTerm::classify_exact_hessian_basin(
            &block,
            &Array1::<f64>::zeros(total_t),
            None,
            total_t,
            "definite-pin",
            None,
        )
        .expect("the spectral route prices the pencil")
        .log_det;
        let resolution_band = (0..dim)
            .map(|index| {
                let tau = block.resolution[index];
                tau / (block.eigenvalues[index] - tau)
            })
            .sum::<f64>();
        let factor_band = dim as f64 * growth * dim as f64 * a_frobenius / a_min;
        let log_det_band = resolution_band + factor_band;
        let log_det_gap = (definite.log_det() - spectral_log_det).abs();
        eprintln!(
            "[F39 pin] log|A| definite {:.15e} spectral {spectral_log_det:.15e} gap {log_det_gap:.3e} \
             band {log_det_band:.3e}",
            definite.log_det()
        );
        assert!(
            spectral_log_det.abs() > 10.0 * log_det_band,
            "magnitude floor: |log|A|| {spectral_log_det:.3e} must clear its band {log_det_band:.3e}"
        );
        assert!(
            log_det_gap <= log_det_band,
            "log|A|: definite and spectral routes differ by {log_det_gap:.3e} > {log_det_band:.3e}"
        );

        let rhs = Array1::from_iter((0..dim).map(|index| uniform(9001 + index as u64)));
        let spectral_solve = block
            .solve_stationarity(&SaeArrowVector {
                t: rhs.slice(s![..total_t]).to_owned(),
                beta: rhs.slice(s![total_t..]).to_owned(),
            })
            .expect("the spectral route solves")
            .step;
        let spectral_x = Array1::from_iter(
            spectral_solve.t.iter().chain(spectral_solve.beta.iter()).copied(),
        );
        let definite_x = definite.solve(rhs.view()).expect("the definite route solves");
        let x_norm = spectral_x.dot(&spectral_x).sqrt();
        let solve_gap = (&definite_x - &spectral_x).dot(&(&definite_x - &spectral_x)).sqrt();
        let solve_band = 2.0 * condition * dim as f64 * growth * x_norm;
        eprintln!("[F39 pin] solve gap {solve_gap:.3e} band {solve_band:.3e} ‖x‖ {x_norm:.3e}");
        assert!(x_norm > 10.0 * solve_band, "magnitude floor on the solve");
        assert!(solve_gap <= solve_band, "A⁻¹rhs: gap {solve_gap:.3e} > {solve_band:.3e}");

        let mut spectral_inverse = Array2::<f64>::zeros((dim, dim));
        for index in 0..dim {
            let w = block.eigenvectors.column(index);
            let weight = 1.0 / block.eigenvalues[index];
            for left in 0..dim {
                for right in 0..dim {
                    spectral_inverse[[left, right]] += weight * w[left] * w[right];
                }
            }
        }
        let inverse_norm = spectral_inverse.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inverse_gap = (&definite.dense_inverse() - &spectral_inverse)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let inverse_band = 2.0 * condition * dim as f64 * growth * inverse_norm;
        eprintln!("[F39 pin] inverse gap {inverse_gap:.3e} band {inverse_band:.3e}");
        assert!(inverse_norm > 10.0 * inverse_band, "magnitude floor on the inverse");
        assert!(inverse_gap <= inverse_band, "A⁻¹: gap {inverse_gap:.3e} > {inverse_band:.3e}");
    }

    /// #2933 F39 — negative controls: every pencil with a direction the spectral route does
    /// not retain positive is refused, whichever block of the count carries it.
    /// * An exact null supported on one row: `A z = 0`, caught by that row's factor.
    /// * A negative direction through the border: every row block definite, the Schur
    ///   complement not. A count over the row blocks alone would miss it.
    /// * A border Schur pivot at zero: every row block definite and the Schur complement
    ///   singular along its flattest direction, so the pencil's smallest `μ` is inside any band.
    /// * A negative direction through the mass carrier: the arrow part definite, `A` not.
    ///   A count that dropped the carrier augmentation would miss it.
    /// * One row block's sign flipped: the miscount mutant's witness. A count that read a
    ///   pivot's magnitude instead of its sign would call it definite.
    /// * A positive direction below the certified shift: refused conservatively, since the
    ///   certificate cannot tell it from one inside its edge.
    #[test]
    fn a_pencil_the_spectral_route_does_not_retain_is_refused_2933_f39() {
        let (system, cache) = evidence();
        let metric = ArrowMetric::Joint(&cache).prepare().expect("the evidence metric prepares");
        let base = operator(&system, 2.0, 0.0);
        let shift = match SaeManifoldTerm::certify_definite_exact_a(&base, &metric)
            .expect("the certificate evaluates")
        {
            Ok(definite) => definite.certificate.shift,
            Err(refusal) => panic!("premise: the base pencil certifies, but: {refusal}"),
        };
        // `A + (μ₀ − μ_z)·(Az)(Az)ᵀ/(zᵀAz)` sets the pencil value along a row-supported `z`
        // (whose `Az` stays inside that row and the border) to `μ₀` in `A`'s own units.
        let along_row = |arrow: &ExactAArrow, target: f64| -> ExactAArrow {
            let dense_a = dense(arrow);
            let dim = dense_a.nrows();
            let mut z = Array1::<f64>::zeros(dim);
            z[0] = 1.0;
            z[1] = 0.5;
            let az = dense_a.dot(&z);
            let curvature = z.dot(&az);
            let metric_curvature = metric.apply(z.view()).expect("Φz").dot(&z);
            let delta = (target * metric_curvature - curvature) / (curvature * curvature);
            let mut rows = arrow.rows.clone();
            let mut border = arrow.border.clone();
            let total_t = ROWS * Q;
            for left in 0..Q {
                for right in 0..Q {
                    rows[0].0[[left, right]] += delta * az[left] * az[right];
                }
                for column in 0..K {
                    rows[0].1[[left, column]] += delta * az[left] * az[total_t + column];
                }
            }
            for left in 0..K {
                for right in 0..K {
                    border[[left, right]] +=
                        delta * az[total_t + left] * az[total_t + right];
                }
            }
            ExactAArrow {
                rows,
                border,
                carriers: arrow.carriers.clone(),
            }
        };
        let mut border_negative = base.rows.clone();
        for (_, tbeta) in &mut border_negative {
            *tbeta *= 20.0;
        }
        let border_negative = ExactAArrow {
            rows: border_negative,
            border: base.border.clone(),
            carriers: Vec::new(),
        };
        let heavy_carrier = {
            let entries: Vec<(usize, f64)> = (0..ROWS).map(|row| (row * Q, 1.0)).collect();
            let u = Array1::from_iter((0..ROWS * Q + K).map(|index| {
                entries
                    .iter()
                    .find(|(position, _)| *position == index)
                    .map_or(0.0, |(_, value)| *value)
            }));
            let curvature = dense(&base).dot(&u).dot(&u);
            let norm_sq = u.dot(&u);
            ExactAArrow {
                rows: base.rows.clone(),
                border: base.border.clone(),
                carriers: vec![(-2.0 * curvature / (norm_sq * norm_sq), entries)],
            }
        };
        // Every row block definite, the border's Schur complement singular along its flattest
        // direction: `A_ββ − λ_min(S)·vvᵀ` with `S = A_ββ − Σᵢ A_βtⁱ(A_ttⁱ)⁻¹A_tβⁱ`, so the
        // border pivot of the count sits at zero, inside any band.
        let border_in_band = {
            let mut schur = base.border.clone();
            for (tt, tbeta) in &base.rows {
                let factor = gam_linalg::triangular::cholesky_factor_in_place(
                    tt.view(),
                    gam_linalg::triangular::CholeskyGuard::FiniteStrict,
                )
                .expect("premise: the base row blocks are definite");
                let solved = gam_linalg::triangular::cholesky_solve_matrix(factor.view(), tbeta.view());
                schur -= &tbeta.t().dot(&solved);
            }
            let (values, vectors) = schur.eigh(Side::Lower).expect("the border Schur decomposes");
            let index = (0..values.len())
                .min_by(|&left, &right| values[left].total_cmp(&values[right]))
                .expect("the border has directions");
            let flattest = values[index];
            let v = vectors.column(index).to_owned();
            let mut border = base.border.clone();
            for left in 0..K {
                for right in 0..K {
                    border[[left, right]] -= flattest * v[left] * v[right];
                }
            }
            ExactAArrow {
                rows: base.rows.clone(),
                border,
                carriers: Vec::new(),
            }
        };
        // The miscount mutant's witness: one row block's sign flipped, so a count that read
        // one pivot's magnitude instead of its sign would call the pencil definite.
        let sign_flipped = {
            let mut rows = base.rows.clone();
            rows[ROWS - 1].0 = -&rows[ROWS - 1].0;
            ExactAArrow {
                rows,
                border: base.border.clone(),
                carriers: Vec::new(),
            }
        };
        let cases = [
            ("row null", along_row(&base, 0.0)),
            ("border-negative", border_negative),
            ("border Schur in band", border_in_band),
            ("carrier-negative", heavy_carrier),
            ("sign-flipped row", sign_flipped),
            ("below the shift", along_row(&base, 0.5 * shift)),
        ];
        for (label, arrow) in cases {
            let block = SaeManifoldTerm::exact_hessian_spectral_block(dense(&arrow), &metric)
                .expect("the spectral route decomposes the pencil");
            let (retained, in_band, negative) = census(&block);
            let smallest = block.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
            eprintln!(
                "[F39 control] {label}: retained {retained} in band {in_band} negative {negative}, \
                 min μ {smallest:.6e}, shift {shift:.3e}"
            );
            assert!(
                smallest <= shift,
                "premise ({label}): the pencil must carry a direction at or below the shift"
            );
            match SaeManifoldTerm::certify_definite_exact_a(&arrow, &metric)
                .expect("the certificate evaluates")
            {
                Ok(definite) => panic!(
                    "{label}: a pencil with min μ {smallest:.3e} must not certify above {:.3e}",
                    definite.certificate.shift
                ),
                Err(refusal) => eprintln!("[F39 control] {label}: refused, {refusal}"),
            }
        }
    }
}

//! ARD (automatic relevance determination) coordinate-precision + latent-block
//! helpers for `SaeManifoldTerm`, split out of `construction.rs` to keep that
//! file under the 10k-line ban gate.
use super::*;
use gam_math::constrained_partition::{
    bingham_sphere_log_partition_and_second_moments,
    normal_interval_log_mass_and_log_precision_score,
};
use gam_math::special::bessel_i0_centered_terms_from_log_abs;
use gam_terms::latent::CoordinatePriorSupport;

/// `(log M(κ, P), d log M / d log κ)`: the log volume one periodic coordinate of period `P`
/// holds under the von Mises energy matched to curvature `κ` at its mode (#2933 F07).
///
/// ```text
///   M(κ, P) = ∫₀ᴾ exp[−(κ/k²)(1 − cos k·u)] du = P·e^{−η}·I0(η),   k = 2π/P,  η = κP²/(2π)²
///   d log M / d log κ = η·(I1(η)/I0(η) − 1)
/// ```
///
/// (substitute `x = k·u`, `du = (P/2π)·dx`, and `∫₀^{2π} e^{η cos x} dx = 2π·I0(η)`).
///
/// One formula covers every curvature, with no switch and no threshold:
///
/// * as `κ → 0`, `M → P`: the coordinate's own volume, which no Laplace factor may exceed;
/// * as `η → ∞`, `M → √(2π/κ)`: the Gaussian Laplace factor, with relative excess
///   `1/(8η) + O(η⁻²)`;
/// * the transition scale `κ ~ (2π/P)²` comes out of the integral itself.
///
/// This is the one owner of a compact coordinate's integral. The ARD prior's periodic
/// normalizer calls it with `κ = α` ([`SaeManifoldTerm::ard_log_partition`]), and a posterior
/// that integrates a periodic coordinate calls it with that coordinate's curvature.
///
/// `η` is formed in log space, because `η` and `κ` can leave the float range while `log M`
/// stays representable. The centered Bessel primitive evaluates `−η + log I0(η)` as one
/// quantity, so its derivative keeps the exact `−½` large-`η` limit after `I1/I0` has rounded
/// to one.
pub(crate) fn circle_log_marginal(log_kappa: f64, period: f64) -> (f64, f64) {
    let log_eta = log_kappa + 2.0 * (period.ln() - std::f64::consts::TAU.ln());
    let (centered_log_i0, _, scaled_derivative) = bessel_i0_centered_terms_from_log_abs(log_eta);
    (period.ln() + centered_log_i0, scaled_derivative)
}

/// `(log M(κ, P), d log M / dκ)` for any real curvature `κ`: the integral of
/// [`circle_log_marginal`] continued through `κ ≤ 0` (#2933 F07).
///
/// A unit-pinned direction of the evidence factor still has its raw curvature, which can
/// be zero or negative. The same energy integrates there without a threshold:
///
/// ```text
///   κ > 0:  log M = log P − η + log I0(η)
///   κ ≤ 0:  log M = log P + |η| + log I0(|η|)          (I0 even, η = κP²/(2π)² ≤ 0)
///   d log M / dκ = (P/2π)²·(I1(η)/I0(η) − 1)          (I1 odd)
/// ```
///
/// so `M` is `P` at `κ = 0` and grows past it as the phase's mode turns into its maximum:
/// the mass of `e^{|η|(1 − cos x)}` sits at the antipode, which the circle holds.
pub(crate) fn circle_log_marginal_signed(kappa: f64, period: f64) -> (f64, f64) {
    let scale = (period / std::f64::consts::TAU).powi(2);
    if kappa > 0.0 {
        let (log_marginal, scaled_derivative) = circle_log_marginal(kappa.ln(), period);
        let eta = kappa * scale;
        // Below `η = 1` the ratio form has no cancellation, and it stays exact where
        // `η` underflows and `η·d/dη` would round to zero.
        let derivative = if eta < 1.0 {
            let (_, ratio, _) = gam_math::special::bessel_i0_centered_terms(eta);
            scale * (ratio - 1.0)
        } else {
            scaled_derivative / kappa
        };
        return (log_marginal, derivative);
    }
    let magnitude = -kappa * scale;
    let (centered_log_i0, ratio, _) = gam_math::special::bessel_i0_centered_terms(magnitude);
    (
        period.ln() + 2.0 * magnitude + centered_log_i0,
        -scale * (1.0 + ratio),
    )
}

/// Per-row log partition of one atom's ARD coordinate prior and its
/// log-precision derivatives (see [`SaeManifoldTerm::ard_log_partition`]).
pub(crate) struct ArdLogPartition {
    /// One log partition per support factor, in axis order.
    pub(crate) per_factor: Vec<f64>,
    /// `∂/∂log α_axis` of the summed log partition, one entry per axis.
    pub(crate) log_precision_gradient: Array1<f64>,
}

impl SaeManifoldTerm {
    /// Per-row log partition of an atom's ARD coordinate prior over the
    /// coordinate's actual support, with its log-precision derivatives.
    ///
    /// Each factor enters paired with the Laplace integration constant
    /// `−½·log 2π` per integrated dimension, which `½·log|A|` leaves out
    /// (#2933 F26):
    ///
    /// * a line axis: `−½ log α`, from the Gaussian `√(2π/α)`;
    /// * a periodic axis: the von Mises partition `log P − η + log I0(η) − ½ log 2π`
    ///   with `η = αP²/(2π)²`, where `P` is the PRIOR period `prior_periods` gives
    ///   (a quotient atom's half-turned axis uses the half period, #2933 F25);
    /// * an interval `[lo, hi]`: `−½ log α + log[Φ(hi√α) − Φ(lo√α)]`. The mass
    ///   factor reaches one only as the interval covers the line. On `[−1, 1]` at
    ///   `α = 1` the log-precision derivative is `−0.1456`, not the line's `−½`
    ///   (#2933 F24).
    /// * an embedded unit sphere `S^(d−1)` holding `d` axes: the Bingham partition
    ///   of `Σ_a ½α_a x_a²` against the surface measure, less `((d−1)/2)·log 2π`,
    ///   with `∂/∂log α_a = −½α_a·E[x_a²]`. Since `‖x‖ = 1`, adding one constant to
    ///   every `α_a` leaves the normalized prior unchanged, and an isotropic
    ///   precision carries no score. The three independent line normalizers this
    ///   replaces scored `−1` per row at `α = 1` (#2933 F24).
    pub(crate) fn ard_log_partition(
        supports: &[CoordinatePriorSupport],
        prior_periods: &[Option<f64>],
        log_alpha: ndarray::ArrayView1<'_, f64>,
        alpha: ndarray::ArrayView1<'_, f64>,
    ) -> Result<ArdLogPartition, String> {
        if prior_periods.len() != log_alpha.len() {
            return Err(format!(
                "ARD log partition: {} prior periods for {} ARD axes",
                prior_periods.len(),
                log_alpha.len()
            ));
        }
        let mut per_factor = Vec::with_capacity(supports.len());
        let mut log_precision_gradient = Array1::<f64>::zeros(log_alpha.len());
        let mut axis = 0;
        for support in supports {
            let width = support.ambient_axes();
            if axis + width > log_alpha.len() {
                return Err(format!(
                    "ARD log partition: support factors span more than the {} ARD axes",
                    log_alpha.len()
                ));
            }
            let periodic = matches!(support, CoordinatePriorSupport::Circle { .. });
            if prior_periods[axis..axis + width].iter().any(|period| period.is_some() != periodic) {
                return Err(format!(
                    "ARD log partition: support {support:?} at axis {axis} disagrees with the \
                     prior periods {:?}",
                    &prior_periods[axis..axis + width]
                ));
            }
            let log_partition = match *support {
                CoordinatePriorSupport::Line => {
                    log_precision_gradient[axis] = -0.5;
                    -0.5 * log_alpha[axis]
                }
                CoordinatePriorSupport::Circle { .. } => {
                    let Some(period) = prior_periods[axis] else {
                        return Err(format!("ARD log partition: periodic axis {axis} has no prior period"));
                    };
                    // The partition over one period is the circle integral of the prior's
                    // own energy `V = (α/k²)(1 − cos k·t)`, `Z(α) = M(α, P)`, which tends to
                    // `½·log(2π/α)` as η → ∞ and is paired the way the line's `½·log 2π`
                    // is. The constants are ρ-independent.
                    let (log_volume, log_curvature_derivative) =
                        circle_log_marginal(log_alpha[axis], period);
                    log_precision_gradient[axis] = log_curvature_derivative;
                    log_volume - 0.5 * std::f64::consts::TAU.ln()
                }
                CoordinatePriorSupport::Interval { lo, hi } => {
                    let root = alpha[axis].sqrt();
                    let (log_mass, score) =
                        normal_interval_log_mass_and_log_precision_score(lo * root, hi * root)
                            .map_err(|error| {
                                format!(
                                    "ARD interval log partition on [{lo}, {hi}] at precision {}: {error}",
                                    alpha[axis]
                                )
                            })?;
                    log_precision_gradient[axis] = -0.5 + score;
                    -0.5 * log_alpha[axis] + log_mass
                }
                CoordinatePriorSupport::Sphere { dim } => {
                    let lambda: Vec<f64> = (axis..axis + width).map(|a| 0.5 * alpha[a]).collect();
                    let (log_z, second_moments) =
                        bingham_sphere_log_partition_and_second_moments(&lambda)
                            .map_err(|error| format!("ARD sphere log partition: {error}"))?;
                    for (offset, (&energy_coefficient, &moment)) in
                        lambda.iter().zip(second_moments.iter()).enumerate()
                    {
                        log_precision_gradient[axis + offset] = -energy_coefficient * moment;
                    }
                    log_z - 0.5 * (dim as f64 - 1.0) * std::f64::consts::TAU.ln()
                }
            };
            per_factor.push(log_partition);
            axis += width;
        }
        if axis != log_alpha.len() {
            return Err(format!(
                "ARD log partition: support factors span {axis} axes but the atom has {} ARD axes",
                log_alpha.len()
            ));
        }
        Ok(ArdLogPartition {
            per_factor,
            log_precision_gradient,
        })
    }

    /// Per-axis period of atom `atom`'s ARD coordinate prior: the kind's
    /// [`SaeAtomBasisKind::ard_axis_periods`] of the coordinate block's wrap
    /// periods, so a quotient atom's half-turned axis carries its deck-invariant
    /// half period (#2933 F25). Every prior consumer (value, gradient, curvature,
    /// traces, IFT channels) reads this seam; `effective_axis_periods` stays the
    /// retraction's wrap period.
    pub(crate) fn ard_axis_periods(&self, atom: usize) -> Vec<Option<f64>> {
        self.atoms[atom]
            .basis_kind()
            .ard_axis_periods(&self.assignment.coords[atom].effective_axis_periods())
    }

    /// [`Self::ard_axis_periods`] for every atom, hoisted out of row loops.
    pub(crate) fn all_ard_axis_periods(&self) -> Vec<Vec<Option<f64>>> {
        (0..self.assignment.coords.len())
            .map(|atom| self.ard_axis_periods(atom))
            .collect()
    }

    /// Embedded unit-sphere factors `(offset, dim)` of every atom's coordinate
    /// block, in the axis order `apply_sae_riemannian_geometry` walks the block's
    /// manifold, hoisted out of row loops.
    pub(crate) fn all_ard_embedded_sphere_factors(&self) -> Vec<Vec<(usize, usize)>> {
        self.assignment
            .coords
            .iter()
            .map(|coord| {
                let mut factors = Vec::new();
                Self::collect_embedded_sphere_factors(coord.manifold(), 0, &mut factors);
                factors
            })
            .collect()
    }

    fn collect_embedded_sphere_factors(
        manifold: &LatentManifold,
        offset: usize,
        factors: &mut Vec<(usize, usize)>,
    ) {
        match manifold {
            LatentManifold::Sphere { dim } => factors.push((offset, *dim)),
            LatentManifold::Product(parts)
            | LatentManifold::ProductWithMetric {
                manifolds: parts, ..
            } => {
                let mut part_offset = offset;
                for part in parts {
                    Self::collect_embedded_sphere_factors(part, part_offset, factors);
                    part_offset += part.ambient_dim(1);
                }
            }
            LatentManifold::Euclidean | LatentManifold::Circle { .. } | LatentManifold::Interval { .. } => {}
        }
    }

    /// Row-local `∂H_tt/∂log α` of one ARD axis that lies on an embedded unit
    /// sphere, as the `q×q` block of a row whose atom block starts at
    /// `block_start`. `None` for an axis on a flat factor (line, circle, interval),
    /// whose derivative stays the slot entry `curvature·e_s e_sᵀ`.
    ///
    /// The assembly writes the ambient prior curvature `h = w·V''` on the axis
    /// slot and the ambient gradient `g = w·V'` into the row, then converts the row
    /// to its Riemannian block (`apply_sae_riemannian_geometry`). On a sphere at
    /// `x`, with `P = I − xxᵀ`, that block is `P H P − (gᵀx)·P + xxᵀ`. Both
    /// operands are degree one in `α`, so `∂/∂log α_a` is `h·P e_a e_aᵀ P −
    /// g_a x_a·P`, not the ambient slot `h·e_a e_aᵀ`. The slot adds the normal
    /// pin's inverse `x_a²` to the trace and drops the connection term (#2933 F24).
    pub(crate) fn ard_sphere_log_precision_derivative(
        sphere_factors: &[(usize, usize)],
        point: &[f64],
        axis: usize,
        block_start: usize,
        q: usize,
        curvature: f64,
        gradient: f64,
    ) -> Option<Array2<f64>> {
        let (offset, dim) = Self::ard_sphere_factor_containing(sphere_factors, axis)?;
        let x = &point[offset..offset + dim];
        let local = axis - offset;
        let start = block_start + offset;
        let mut derivative = Array2::<f64>::zeros((q, q));
        for i in 0..dim {
            let tangent_i = Self::sphere_tangent_of_axis(x, local, i);
            for j in 0..dim {
                let tangent_j = Self::sphere_tangent_of_axis(x, local, j);
                let projector = (if i == j { 1.0 } else { 0.0 }) - x[i] * x[j];
                derivative[[start + i, start + j]] =
                    curvature * tangent_i * tangent_j - gradient * x[local] * projector;
            }
        }
        Some(derivative)
    }

    /// The embedded unit-sphere factor `(offset, dim)` holding `axis`, if any.
    pub(crate) fn ard_sphere_factor_containing(
        sphere_factors: &[(usize, usize)],
        axis: usize,
    ) -> Option<(usize, usize)> {
        sphere_factors
            .iter()
            .copied()
            .find(|&(offset, dim)| offset <= axis && axis < offset + dim)
    }

    /// Component `i` of the tangent projection `P e_local = e_local − x_local·x` at
    /// the unit vector `x`.
    pub(crate) fn sphere_tangent_of_axis(x: &[f64], local: usize, i: usize) -> f64 {
        (if i == local { 1.0 } else { 0.0 }) - x[local] * x[i]
    }

    /// Validate the ARD table against this term's atom geometry and materialize
    /// each physical precision exactly once. This is the structural choke point
    /// shared by assembly, value, traces, exact-Hessian, and IFT channels.
    ///
    /// #2822 — every coordinate atom carries a full block. The ARD prior is the
    /// proper coordinate prior: without it a row's coordinate posterior is improper,
    /// and the criterion has no lower bound along that coordinate. So an empty block
    /// is refused here, where every criterion path first reads the table, rather than
    /// read as a prior that is switched off.
    pub(crate) fn validated_ard_precisions(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        if rho.log_ard.len() != self.k_atoms() {
            return Err(format!(
                "ARD rho has {} atom blocks but term has {} atoms",
                rho.log_ard.len(),
                self.k_atoms()
            ));
        }
        for (atom, coordinate) in self.assignment.coords.iter().enumerate() {
            let stored = rho.log_ard[atom].len();
            let dimension = coordinate.latent_dim();
            if stored != dimension {
                return Err(format!(
                    "ARD rho atom {atom} has {stored} axes but its coordinate has latent \
                     dimension {dimension}: every coordinate atom carries a full log_ard block, \
                     because the ARD prior is the proper coordinate prior its rows' posteriors \
                     need (#2822)"
                ));
            }
        }
        rho.ard_precisions()
    }

    /// Rows on which each atom's coordinate is a variable of the model (#2933 F27).
    ///
    /// A hard-TopK row block holds only the selected atoms' coordinates
    /// ([`SaeRowLayout::from_topk_gates`]). So `½·log|A|` integrates a coordinate only
    /// on the rows that select its atom, and the inner solve never moves it on the
    /// others. The ARD prior's energy, its log partition and their precision
    /// derivatives are priced on that same slot set, as the support-sparse route
    /// prices them on its active slots. Entry `k` lists atom `k`'s selecting rows in
    /// row order. Every other assignment family holds every atom's coordinate on
    /// every row, which is `None`.
    pub(crate) fn coordinate_prior_rows(&self) -> Result<Option<Vec<Vec<usize>>>, String> {
        let AssignmentMode::TopK { k } = self.assignment.mode else {
            return Ok(None);
        };
        let n = self.n_obs();
        let layout = SaeRowLayout::from_topk_gates(
            &self.assignments_all_parallel(n)?,
            k,
            self.assignment
                .coords
                .iter()
                .map(|coord| coord.latent_dim())
                .collect(),
            self.assignment.coord_offsets(),
        )?;
        let mut rows = vec![Vec::new(); self.k_atoms()];
        for (row, active) in layout.active_atoms.iter().enumerate() {
            for &atom in active {
                rows[atom].push(row);
            }
        }
        Ok(Some(rows))
    }

    /// Per-atom, per-axis coordinate sum-of-squares `‖t_kj‖² = Σ_i t_{i,k,j}²`.
    ///
    /// This is the data-fit sufficient statistic for the ARD precision update
    /// (the numerator-side `‖t‖²` of the deleted `α = n/‖t‖²` rule). Returned
    /// per atom as an `Array1` of length `d_k`. The sum runs over the rows that hold
    /// the coordinate ([`Self::coordinate_prior_rows`]), the slot set `ard_value`
    /// prices and `ard_inverse_traces` sums `tr H⁻¹` over.
    ///
    /// On a *periodic* (Circle) axis the relevant statistic is the von-Mises
    /// energy-equivalent `Σ_i 2/α·V(t_i) = Σ_i (2/κ²)(1−cos κ t_i)` (independent
    /// of α), so that `½·α·sumsq == Σ_i V(t_i)` matches `ard_value`. This keeps
    /// the Mackay/Fellner–Schall fixed point `α ← n / (sumsq + tr H⁻¹)`
    /// consistent with the actual periodic prior energy rather than the
    /// origin-dependent raw `t²`.
    pub(crate) fn ard_coord_sumsq(&self) -> Result<Vec<Array1<f64>>, String> {
        // Horvitz–Thompson row weighting: the `‖t‖²` sufficient statistic is the
        // numerator of the same `α ← n_eff / (Σ sq_equiv + tr H⁻¹)` fixed point the
        // (now weight-aware) `ard_value` energy defines, so it MUST carry the SAME
        // per-row inclusion weight `wᵢ` — else the subsampled MacKay/Fellner–Schall
        // step ranks a different precision than the criterion's energy. `None` ⇒
        // `w_row = 1`, bit-for-bit the historical sum.
        let row_w = self.row_loss_weights.as_deref();
        let prior_rows = self.coordinate_prior_rows()?;
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let periods = self.ard_axis_periods(atom_idx);
            let atom_rows = prior_rows.as_ref().map(|rows| rows[atom_idx].as_slice());
            let slots = atom_rows.map_or(coord.n_obs(), <[usize]>::len);
            let mut sq = Array1::<f64>::zeros(d);
            for slot in 0..slots {
                let row = atom_rows.map_or(slot, |rows| rows[slot]);
                let w_row = row_w.map_or(1.0, |w| w[row]);
                let t = coord.row(row);
                for axis in 0..d {
                    // `sq_equiv` is independent of `alpha`; pass 1.0.
                    sq[axis] += w_row * ArdAxisPrior::eval(1.0, t[axis], periods[axis]).sq_equiv;
                }
            }
            out.push(sq);
        }
        Ok(out)
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹) =
    /// Σ_i [(H⁻¹)_tt]_{(i,k,j),(i,k,j)}` from the converged factor cache.
    ///
    /// `cache.latent_block_inverse_diagonal()` returns the diagonal of the
    /// latent block `(H⁻¹)_tt` in the cache's compact per-row `delta_t`
    /// layout (length `row_offsets[N]`). A compact hard-TopK row contains only
    /// the selected atoms' coordinate axes; a dense row contains assignment
    /// coordinates followed by all atom-coordinate axes. This routine
    /// sums those diagonal entries over the coord positions belonging to each
    /// `(atom k, axis j)` across all observation rows where atom `k` is active.
    ///
    /// `self.last_row_layout` must be the layout from the *same* assemble that
    /// produced `cache`:
    /// - `Some(layout)`: compact exact hard-TopK support. For row `i`, atom `k`'s position in the
    ///   active list gives its compact coord-block start `coord_starts[i][pos]`;
    ///   inactive atoms contribute 0 (the prior dominates there anyway).
    /// - `None`: dense full-support layout, uniform row dim
    ///   `q = assignment_dim + Σ d_k`; atom `k`'s coord block sits at the
    ///   fixed full-row offset `coord_offsets[k]` after the assignment chart.
    ///
    /// This `tr_kj(H⁻¹)` is exactly the posterior-variance term the deleted
    /// `α = n/‖t‖²` rule dropped; the corrected Mackay/Fellner-Schall fixed
    /// point is `α_new = n / (‖t_kj‖² + tr_kj(H⁻¹))`.
    ///
    /// The diagonal is exact at every `K`: `latent_block_inverse_diagonal` pays
    /// `K` Schur applies once plus each coordinate's touched border columns, the
    /// order of the Schur factorization the cache already holds (#2900 row 6.18).
    pub(crate) fn ard_inverse_traces(
        &self,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        let inv_diag = cache.latent_block_inverse_diagonal()?;
        Ok(self.accumulate_latent_inverse_diagonal(cache, &inv_diag, |_, _, _| 1.0))
    }

    /// Sum the latent inverse diagonal into per-`(atom, axis)` groups, one
    /// dimensionless `slot_factor(row, atom, axis)` per slot.
    ///
    /// The single source of truth for walking the arrow's latent layout — both
    /// the compact hard-TopK support and the dense full-support layout — so the
    /// trace [`Self::ard_inverse_traces`] reads off the diagonal (with
    /// `factor ≡ 1`) visits exactly the slots the layout declares.
    ///
    /// Horvitz–Thompson row weight, IDENTICAL to `ard_coord_sumsq`'s numerator
    /// weighting: the posterior-variance trace `tr H⁻¹` is the OTHER half of
    /// the `α ← n_eff / (Σ wᵢ t̂ᵢ² + Σ wᵢ (H⁻¹)ᵢᵢ)` MacKay/Fellner–Schall
    /// fixed point, so a retained row standing in for `wᵢ` rows must
    /// contribute its posterior variance `wᵢ` times too — else the α-step's
    /// denominator uses a different inclusion measure than its numerator and
    /// `n_eff`. Commit 4862e8355 weighted value/sumsq/gradient/curvature
    /// "together" but missed this trace channel. `None` ⇒ `wᵢ = 1`,
    /// bit-for-bit the historical unweighted sum.
    fn accumulate_latent_inverse_diagonal<F>(
        &self,
        cache: &ArrowFactorCache,
        inv_diag: &Array1<f64>,
        mut slot_factor: F,
    ) -> Vec<Array1<f64>>
    where
        F: FnMut(usize, usize, usize) -> f64,
    {
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let row_w = self.row_loss_weights.as_deref();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let row_base = cache.row_offsets[row];
            let w_row = row_w.map_or(1.0, |w| w[row]);
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = starts[pos];
                        for axis in 0..d {
                            traces[k][axis] += slot_factor(row, k, axis)
                                * w_row
                                * inv_diag[row_base + block_start + axis];
                        }
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        let d = self.assignment.coords[k].latent_dim();
                        let block_start = coord_offsets[k];
                        for axis in 0..d {
                            traces[k][axis] += slot_factor(row, k, axis)
                                * w_row
                                * inv_diag[row_base + block_start + axis];
                        }
                    }
                }
            }
        }
        traces
    }

    /// Per-atom, per-axis posterior-variance trace `tr_kj(H⁻¹)` — the SAME
    /// quantity [`Self::ard_inverse_traces`] returns — from the #2080 SHARED
    /// selected-inverse bundle instead of the dense Schur factor
    /// (`full_inverse_apply` / `latent_block_inverse_diagonal`). The massive-`K`
    /// surrogate-lane replacement that removes the last dense `S⁻¹` from the ARD
    /// Fellner–Schall denominator.
    ///
    /// # Reformulation (arrow selected-inverse, no dense `S⁻¹`)
    ///
    /// For an arrow `H = [[A (⊕_i A_i), B], [Bᵀ, C]]` with reduced Schur
    /// complement `S`, the per-row latent block is exactly
    /// `(H⁻¹)_{t_i t_i} = A_i⁻¹ + G_i S⁻¹ G_iᵀ`, `A_i = H_tt^(i)` (the per-row
    /// `undamped_factor`), `B_i = H_tβ^(i)` (via `apply_htbeta_row`),
    /// `G_i = A_i⁻¹ B_i`. So the per-slot diagonal the ARD denominator sums splits
    /// into a ROW-LOCAL exact term `(A_i⁻¹)[s,s]` and a border term
    /// `(G_i S⁻¹ G_iᵀ)[s,s] = g_sᵀ S⁻¹ g_s`, `g_s = G_iᵀ e_s`. Summed over rows
    /// for a FIXED `(atom k, axis a)` — the group the ARD α-fixed-point actually
    /// needs — the border piece is a trace `Σ_i g_{s_i}ᵀ S⁻¹ g_{s_i} =
    /// tr(S⁻¹ M_{ka})`, `M_{ka} = Σ_i g_{s_i} g_{s_i}ᵀ`, estimated off the shared
    /// bundle `(z_j, S⁻¹ z_j)` by
    ///   `tr(S⁻¹ M_{ka}) = (1/m) Σ_j Σ_i (g_{s_i}ᵀ S⁻¹ z_j)(g_{s_i}ᵀ z_j)
    ///                   = (1/m) Σ_j Σ_i s_ij[s_i]·w_ij[s_i]`,
    /// with `w_ij = G_i z_j = A_i⁻¹ B_i z_j` and `s_ij = G_i S⁻¹ z_j =
    /// A_i⁻¹ B_i (S⁻¹ z_j)` — both per-row `t`-space vectors from
    /// `apply_htbeta_row` + a per-row Cholesky solve. The final `H_βt` never
    /// appears: it is absorbed by contracting against `S⁻¹ z_j`. Everything is
    /// sourced from the cache (`undamped_factor` + `apply_htbeta_row`) plus the
    /// bundle — no `ArrowSchurSystem`, no dense `S⁻¹`.
    ///
    /// `probes` and `sinv_probes` are the surrogate lane's frozen `(z_j, S⁻¹ z_j)`
    /// pairs (each length `cache.k`, the reduced-Schur border dim); with
    /// full-basis probes `√k·e_j` the Hutchinson average is exact, which the FD
    /// gate exploits to assert equality with [`Self::ard_inverse_traces`]. HT
    /// row-weighting matches `ard_inverse_traces` bit-for-bit (both diagonal parts
    /// carry `w_row`; `None` ⇒ `w_row = 1`).
    pub(crate) fn ard_inverse_traces_from_probes(
        &self,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
    ) -> Result<Vec<Array1<f64>>, String> {
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(format!(
                "ard_inverse_traces_from_probes: need matching non-empty probe/solve \
                 bundles, got {m} probes and {} solves",
                sinv_probes.len()
            ));
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, v) in set.iter().enumerate() {
                if v.len() != k_border {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: {label} {j} has length {} != border \
                         dim {k_border}",
                        v.len()
                    ));
                }
            }
        }
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let row_w = self.row_loss_weights.as_deref();
        let inv_m = 1.0 / (m as f64);
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .map(|c| Array1::<f64>::zeros(c.latent_dim()))
            .collect();
        for row in 0..n {
            let q = cache.row_dims[row];
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let factor = cache.undamped_factor(row);
            // A_i⁻¹ diagonal (row-local, exact): solve A_i e_s = e_s per local slot.
            let mut a_inv_diag = Array1::<f64>::zeros(q);
            let mut e_s = Array1::<f64>::zeros(q);
            for s in 0..q {
                e_s.fill(0.0);
                e_s[s] = 1.0;
                let col = cholesky_solve_vector(factor, e_s.view());
                a_inv_diag[s] = col[s];
            }
            // Per-probe border vectors w_ij = A_i⁻¹ B_i z_j and s_ij = A_i⁻¹ B_i
            // (S⁻¹ z_j), both row-local `t`-space (length q).
            let mut w_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut s_probes: Vec<Array1<f64>> = Vec::with_capacity(m);
            let mut b_tmp = Array1::<f64>::zeros(q);
            for j in 0..m {
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, probes[j].view(), &mut b_tmp) {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: H_tβ^({row}) probe apply failed"
                    ));
                }
                w_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
                b_tmp.fill(0.0);
                if !cache.apply_htbeta_row(row, sinv_probes[j].view(), &mut b_tmp) {
                    return Err(format!(
                        "ard_inverse_traces_from_probes: H_tβ^({row}) solve apply failed"
                    ));
                }
                s_probes.push(cholesky_solve_vector(factor, b_tmp.view()));
            }
            // Per-slot diagonal = (A_i⁻¹)[s,s] + (1/m) Σ_j w_ij[s]·s_ij[s]; sum into
            // the owning (atom, axis) trace exactly as `ard_inverse_traces` does.
            let accumulate = |k: usize, block_start: usize, traces: &mut Vec<Array1<f64>>| {
                let d = self.assignment.coords[k].latent_dim();
                for axis in 0..d {
                    let s = block_start + axis;
                    let mut border = 0.0_f64;
                    for j in 0..m {
                        border += w_probes[j][s] * s_probes[j][s];
                    }
                    traces[k][axis] += w_row * (a_inv_diag[s] + inv_m * border);
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        accumulate(k, starts[pos], &mut traces);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        accumulate(k, coord_offsets[k], &mut traces);
                    }
                }
            }
        }
        Ok(traces)
    }

    pub(crate) fn ard_log_precision_explicit_derivatives(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Vec<Array1<f64>>, String> {
        self.assignment.validate_rho_domain(rho)?;
        let ard_precisions = self.validated_ard_precisions(rho)?;
        // HT row weighting: this is the ρ-derivative of `ard_value` (the `explicit`
        // outer-gradient channel), so it carries the identical per-row inclusion
        // weight on the energy and the identical slot count on the normalizer as the
        // value — otherwise the analytic gradient desyncs from the criterion value.
        // Both run over the rows that hold the coordinate (#2933 F27,
        // `Self::coordinate_prior_rows`).
        let row_w = self.row_loss_weights.as_deref();
        let prior_rows = self.coordinate_prior_rows()?;
        let mut out = Vec::with_capacity(self.k_atoms());
        for (atom_idx, coord) in self.assignment.coords.iter().enumerate() {
            let d = coord.latent_dim();
            let mut atom_out = Array1::<f64>::zeros(rho.log_ard[atom_idx].len());
            if rho.log_ard[atom_idx].is_empty() {
                out.push(atom_out);
                continue;
            }
            let periods = self.ard_axis_periods(atom_idx);
            // The log partition over the coordinate's support; its derivative is
            // the normalizer channel of `∂ ard_value/∂log α`.
            let partition = Self::ard_log_partition(
                &coord.effective_prior_supports(),
                &periods,
                rho.log_ard[atom_idx].view(),
                ard_precisions[atom_idx].view(),
            )?;
            let atom_rows = prior_rows.as_ref().map(|rows| rows[atom_idx].as_slice());
            let slots = atom_rows.map_or(coord.n_obs(), <[usize]>::len);
            for axis in 0..d {
                let alpha = ard_precisions[atom_idx][axis];
                let period = periods[axis];
                let mut energy_deriv = 0.0_f64;
                for slot in 0..slots {
                    let row = atom_rows.map_or(slot, |rows| rows[slot]);
                    let w_row = row_w.map_or(1.0, |w| w[row]);
                    let t = coord.row(row)[axis];
                    energy_deriv += w_row * ArdAxisPrior::eval(alpha, t, period).value;
                }
                atom_out[axis] =
                    energy_deriv + slots as f64 * partition.log_precision_gradient[axis];
            }
            out.push(atom_out);
        }
        Ok(out)
    }

    /// `operator` (#2515) names WHICH curvature the `∂H/∂log α` operand is:
    /// `Majorizer` differentiates `B`'s PSD-clamped `α·s_{τ₀}(cos κt)`,
    /// `ExactObservedInformation` differentiates `A`'s unmajorized `α·cos κt`.
    /// It must agree with the operator whose inverse `solver` factors, which is
    /// why every trace entry point takes it explicitly rather than defaulting.
    #[cfg(test)]
    pub(crate) fn ard_log_precision_hessian_trace(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        solver: &DeflatedArrowSolver<'_>,
        operator: EvidenceOperator,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let ard_precisions = self
            .validated_ard_precisions(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        // RAW selected-inverse diagonal: the per-axis diagonal contraction uses
        // the DEFLATED inverse; the full kept-subspace + rotation deflation
        // correction `tr(inv_vv·(D − DΦ[D]))` is subtracted per (row, axis)
        // afterwards via the Daleckii–Krein helper. An ARD ρ-component
        // `(atom k, axis)` on a flat factor differentiates a SINGLE coordinate-slot
        // diagonal entry, so its `D` is the rank-one `hess·e_s e_sᵀ` at that local
        // slot `s`. On an embedded sphere `D` is the Riemannian block derivative
        // (`ard_sphere_log_precision_derivative`), contracted against the row's
        // whole selected-inverse block (#2933 F24).
        let inv_diag = solver
            .latent_inverse_diagonal()
            .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
        // HT row weighting: the assembled per-row ARD curvature `∂H/∂logα` is scaled
        // by the inclusion weight `wᵢ` (see the assembly seam), and `inv_diag` = the
        // diagonal of `H⁻¹` already reflects that w-scaled `H`. So each row's trace
        // contribution `½·(H⁻¹)_ss·(wᵢ·hess)` must carry the SAME `wᵢ` here, or the
        // ½log|H| gradient desyncs from the assembled Hessian on the subsample.
        // `None` ⇒ `w_row = 1`, bit-for-bit the historical trace.
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let total_t = cache.delta_t_len();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        // Hoisted RHS scratch reused across every (row, col) solve. Setting and
        // clearing a SINGLE entry per column is O(1); a fresh
        // `Array1::zeros(total_t)` memsets total_t≈n·q slots per inner iteration
        // (O(n) per col ⇒ O(n²) redundant zeroing across the block build).
        let mut rhs_t_scratch = Array1::<f64>::zeros(total_t);
        let rhs_beta_zero = Array1::<f64>::zeros(cache.k);
        let sphere_factors = self.all_ard_embedded_sphere_factors();
        for row in 0..n {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let row_base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            let row_atoms: Vec<(usize, usize)> = match self.last_row_layout {
                Some(ref layout) => layout.active_atoms[row]
                    .iter()
                    .copied()
                    .zip(layout.coord_starts[row].iter().copied())
                    .collect(),
                None => (0..self.k_atoms()).map(|k| (k, coord_offsets[k])).collect(),
            };
            let sphere_row = row_atoms
                .iter()
                .any(|&(k, _)| !rho.log_ard[k].is_empty() && !sphere_factors[k].is_empty());
            // Per-row selected-inverse t-block, built once (only when deflated, or
            // when a sphere axis contracts a whole block).
            let inv_vv = if !Self::row_deflation_is_live(dirs, spectrum) && !sphere_row {
                None
            } else {
                let mut m = Array2::<f64>::zeros((q, q));
                for col in 0..q {
                    rhs_t_scratch[row_base + col] = 1.0;
                    let solved = solver
                        .solve(rhs_t_scratch.view(), rhs_beta_zero.view())
                        .map_err(|err| ArrowSchurError::SchurFactorFailed { reason: err })?;
                    rhs_t_scratch[row_base + col] = 0.0;
                    for r in 0..q {
                        m[[r, col]] = solved.t[row_base + r];
                    }
                }
                Some(m)
            };
            // Correction for one local coordinate slot `s` with curvature `hess`.
            let slot_correction = |s: usize, hess: f64| -> f64 {
                let Some(iv) = inv_vv.as_ref() else {
                    return 0.0;
                };
                if s >= q || hess == 0.0 {
                    return 0.0;
                }
                let mut d = Array2::<f64>::zeros((q, q));
                d[[s, s]] = hess;
                Self::deflation_block_correction(iv, &d, dirs, spectrum)
            };
            for (k, block_start) in row_atoms {
                if rho.log_ard[k].is_empty() {
                    continue;
                }
                let coord = &self.assignment.coords[k];
                let point = coord.row(row);
                for axis in 0..coord.latent_dim() {
                    let alpha = ard_precisions[k][axis];
                    let prior = ArdAxisPrior::eval(alpha, point[axis], ard_axis_periods[k][axis]);
                    let hess = w_row * prior.log_precision_curvature(operator);
                    if let Some(derivative) = Self::ard_sphere_log_precision_derivative(
                        &sphere_factors[k],
                        point,
                        axis,
                        block_start,
                        q,
                        hess,
                        w_row * prior.grad,
                    ) {
                        let Some(inverse) = inv_vv.as_ref() else {
                            return Err(ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "ard_log_precision_hessian_trace: row {row} holds a sphere ARD \
                                     axis but its selected-inverse block was not built"
                                ),
                            });
                        };
                        traces[k][axis] += 0.5
                            * ((inverse * &derivative).sum()
                                - Self::deflation_block_correction(
                                    inverse,
                                    &derivative,
                                    dirs,
                                    spectrum,
                                ));
                        continue;
                    }
                    let s = block_start + axis;
                    traces[k][axis] += 0.5 * inv_diag[row_base + s] * hess;
                    traces[k][axis] -= 0.5 * slot_correction(s, hess);
                }
            }
        }
        Ok(traces)
    }

    /// Per-atom, per-axis `½ tr(H⁻¹ ∂H/∂logα_{kj})` — the ARD ½log|H| ρ-gradient
    /// channel [`Self::ard_log_precision_hessian_trace`] computes — from the #2080
    /// SHARED selected-inverse bundle instead of the dense `DeflatedArrowSolver`
    /// (`latent_inverse_diagonal` + per-column `solve`). The massive-lane /
    /// eventual-dense-cache-retirement replacement for the last dense `S⁻¹` in the
    /// analytic outer ρ-gradient's ARD block.
    ///
    /// Each ARD component differentiates ONE coordinate-slot diagonal entry, so the
    /// trace is `Σ_{row, slot s(k,j)} ½·(H⁻¹)_tt[s,s]·(w_row·hess)`. The diagonal is
    /// the SAME per-row arrow selected-inverse the ARD posterior-variance trace uses
    /// (`(A_i⁻¹)[s,s]` row-local + border `(1/m)Σ_j w_ij[s]·s_ij[s]` off the bundle,
    /// see [`Self::ard_inverse_traces_from_probes`] for the derivation), matching
    /// `solver.latent_inverse_diagonal()` on the PLAIN (undeflated) selected inverse
    /// to solve precision.
    ///
    /// # Deflation is priced, not refused (#2712)
    ///
    /// The dense path's Daleckii–Krein correction `−½ tr(inv_vv·(D − DΦ[D]))`
    /// needs the DEFLATED per-row inverse block, and this lane has it: `A_i` is
    /// `cache.undamped_factor(i)`, the Cholesky of the spectrally CONDITIONED
    /// `Φ(H_tt^(i))`, and the reduced Schur behind the bundle is that same
    /// conditioned arrow's — so `A_i⁻¹ + G_i S⁻¹ G_iᵀ` IS the deflated block
    /// ([`row_selected_inverse_from_probes`]). This lane used to hard-refuse
    /// deflated rows on the stated grounds that the reconstruction was the
    /// UNdeflated block; that was a misreading of `undamped_factor`, and the
    /// correction's remaining operands (`deflated_row_directions`,
    /// `deflation_row_spectra`, and the per-slot diagonal `D` built below) never
    /// involved `S⁻¹` at all. Deflated and undeflated rows now take the same
    /// route, and the correction is identically zero on a PD row.
    // #2080 analytic-gradient cluster channel: wired into
    // `analytic_outer_rho_gradient_components_with_bundle`'s `Some`-bundle branch
    // (the all-or-nothing selected-inverse cluster, alongside the from-probes
    // smoothness EDF), dormant until the analytic-gradient routing flips (every
    // caller passes `None` today). That real call site is the non-test consumer,
    // so this matches its `pub(crate)` sibling `ard_inverse_traces_from_probes`.
    pub(crate) fn ard_log_precision_hessian_trace_from_probes(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        probes: &[Array1<f64>],
        sinv_probes: &[Array1<f64>],
        operator: EvidenceOperator,
    ) -> Result<Vec<Array1<f64>>, ArrowSchurError> {
        self.assignment
            .validate_rho_domain(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let ard_precisions = self
            .validated_ard_precisions(rho)
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let m = probes.len();
        if m == 0 || sinv_probes.len() != m {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "ard_log_precision_hessian_trace_from_probes: need matching non-empty \
                     probe/solve bundles, got {m} probes and {} solves",
                    sinv_probes.len()
                ),
            });
        }
        let k_border = cache.k;
        for (label, set) in [("probe", probes), ("solve", sinv_probes)] {
            for (j, v) in set.iter().enumerate() {
                if v.len() != k_border {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "ard_log_precision_hessian_trace_from_probes: {label} {j} has length \
                             {} != border dim {k_border}",
                            v.len()
                        ),
                    });
                }
            }
        }
        let row_w = self.row_loss_weights.as_deref();
        let n = self.n_obs();
        let coord_offsets = self.assignment.coord_offsets();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self.all_ard_axis_periods();
        let mut traces: Vec<Array1<f64>> = self
            .assignment
            .coords
            .iter()
            .enumerate()
            .map(|(k, c)| {
                if rho.log_ard[k].is_empty() {
                    Array1::<f64>::zeros(0)
                } else {
                    Array1::<f64>::zeros(c.latent_dim())
                }
            })
            .collect();
        // #2915 — a clamp-basin price moves with the clamp itself, which the exact
        // operator's `α·cos κt` does not carry.
        let clamp = if operator.is_exact_a() {
            Some(
                self.materialize_ard_concave_clamp_diagonal_for_rows(rho, &cache.row_dims)
                    .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?,
            )
        } else {
            None
        };
        // #2915 — so does a reduced-Schur clamp-basin price.
        // The border clamp is the decoder priors' remainder at the unit penalty scale of
        // the full evidence system this lane factors.
        let border_remainder = if clamp.is_some() && cache.beta_schur_conditioning.is_some() {
            self.decoder_prior_border_remainder_op(cache.k, 1.0)
                .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?
        } else {
            None
        };
        let beta_price = match clamp.as_ref() {
            Some(clamp) => Self::beta_schur_clamp_basin_price_weights(
                cache,
                clamp.view(),
                border_remainder.as_ref().map(|op| op as &dyn BetaPenaltyOp),
            )
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?,
            None => None,
        };
        let sphere_factors = self.all_ard_embedded_sphere_factors();
        for row in 0..n {
            let w_row = row_w.map_or(1.0, |w| w[row]);
            let q = cache.row_dims[row];
            // The DEFLATED row-block selected inverse from the shared bundle
            // (#2712). The `t–β` block is not contracted here, so it is not built.
            let (mut inv_vv, _) = row_selected_inverse_from_probes(
                cache,
                row,
                probes,
                sinv_probes,
                false,
                "ard_log_precision_hessian_trace_from_probes",
            )
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
            if let Some(weights) = beta_price.as_ref() {
                inv_vv += &weights[row].0;
            }
            let inv_diag_local = inv_vv.diag().to_owned();
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let spectrum = cache
                .deflation_row_spectra
                .get(row)
                .and_then(Option::as_ref);
            let row_base = cache.row_offsets[row];
            let price = clamp.as_ref().and_then(|clamp| {
                Self::clamp_basin_price_weights(
                    &inv_vv,
                    clamp.slice(ndarray::s![row_base..row_base + q]),
                    spectrum,
                )
            });
            // Correction for one local coordinate slot `s` with curvature `hess`,
            // identical to the dense sibling's `slot_correction`.
            let slot_correction = |s: usize, hess: f64| -> f64 {
                if !Self::row_deflation_is_live(dirs, spectrum) || s >= q || hess == 0.0 {
                    return 0.0;
                }
                let mut d = Array2::<f64>::zeros((q, q));
                d[[s, s]] = hess;
                Self::deflation_block_correction(&inv_vv, &d, dirs, spectrum)
            };
            let accumulate = |k: usize, block_start: usize, traces: &mut Vec<Array1<f64>>| {
                if rho.log_ard[k].is_empty() {
                    return;
                }
                let coord = &self.assignment.coords[k];
                let point = coord.row(row);
                for axis in 0..coord.latent_dim() {
                    let alpha = ard_precisions[k][axis];
                    let prior = ArdAxisPrior::eval(alpha, point[axis], ard_axis_periods[k][axis]);
                    let hess = w_row * prior.log_precision_curvature(operator);
                    let s = block_start + axis;
                    // A sphere axis contracts its Riemannian block derivative against
                    // the whole row block (#2933 F24); a flat axis its slot entry.
                    let sphere_derivative = Self::ard_sphere_log_precision_derivative(
                        &sphere_factors[k],
                        point,
                        axis,
                        block_start,
                        q,
                        hess,
                        w_row * prior.grad,
                    );
                    match sphere_derivative.as_ref() {
                        Some(derivative) => {
                            traces[k][axis] += 0.5 * (&inv_vv * derivative).sum();
                            if Self::row_deflation_is_live(dirs, spectrum) {
                                traces[k][axis] -= 0.5
                                    * Self::deflation_block_correction(
                                        &inv_vv, derivative, dirs, spectrum,
                                    );
                            }
                        }
                        None => {
                            traces[k][axis] += 0.5 * inv_diag_local[s] * hess;
                            traces[k][axis] -= 0.5 * slot_correction(s, hess);
                        }
                    }
                    if let (Some((explicit, response)), Some(clamp)) = (price.as_ref(), clamp.as_ref())
                    {
                        if s < q {
                            // `∂E/∂ρ_ard` at slot `s` is the ARD clamp itself: degree one in `α`.
                            let response_trace = match sphere_derivative.as_ref() {
                                Some(derivative) => (response * derivative).sum(),
                                None => response[[s, s]] * hess,
                            };
                            traces[k][axis] +=
                                0.5 * (explicit[s] * clamp[row_base + s] + response_trace);
                        }
                    }
                    if let (Some(weights), Some(clamp)) = (beta_price.as_ref(), clamp.as_ref()) {
                        if s < q {
                            // The reduced-Schur basin prices read the same ARD clamp.
                            traces[k][axis] += 0.5 * weights[row].1[s] * clamp[row_base + s];
                        }
                    }
                }
            };
            match self.last_row_layout {
                Some(ref layout) => {
                    let active = &layout.active_atoms[row];
                    let starts = &layout.coord_starts[row];
                    for (pos, &k) in active.iter().enumerate() {
                        accumulate(k, starts[pos], &mut traces);
                    }
                }
                None => {
                    for k in 0..self.k_atoms() {
                        accumulate(k, coord_offsets[k], &mut traces);
                    }
                }
            }
        }
        Ok(traces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Periods the pins exercise: the harmonic basis's unit period, a quotient atom's half
    /// period (#2933 F25), and a period in radians.
    const PERIODS: [f64; 3] = [1.0, 0.5, TAU];

    /// `(M, E[η(1 − cos k·u)])` for the energy `(κ/k²)(1 − cos k·u)` by the periodic trapezoid
    /// rule on `nodes` equispaced points of one period.
    ///
    /// On a full period the rule's only error is aliasing: `e^{η cos x} = I0(η) + 2Σₙ Iₙ(η)cos nx`,
    /// and `nodes` points integrate every `cos nx` exactly unless `nodes` divides `n`, so the
    /// relative error is `2Σₘ I_{m·nodes}(η)/I0(η)`, below `exp(−nodes²/(2η))` for `nodes > η` and
    /// below `(η/2)^nodes/nodes!` for `η ≤ 1`.
    fn trapezoid_circle_moments(kappa: f64, period: f64, nodes: usize) -> (f64, f64) {
        let k = TAU / period;
        let eta = kappa / (k * k);
        let mut mass = 0.0_f64;
        let mut energy = 0.0_f64;
        for node in 0..nodes {
            let x = TAU * node as f64 / nodes as f64;
            let e = eta * (1.0 - x.cos());
            let weight = (-e).exp();
            mass += weight;
            energy += e * weight;
        }
        (mass * period / nodes as f64, energy / mass)
    }

    /// The trapezoid node count and the relative rounding budget of the pins below. `γ_n =
    /// nε/(1 − nε)` bounds the positive sums. Each summand carries the absolute error `2ηε` of
    /// `η(1 − cos x)` plus `ε` of its own `exp`, so the mass is good to `(3η + 1)ε` relative
    /// (every summand's exponent is at most `2η`).
    const NODES: usize = 4096;

    fn gamma(n: usize) -> f64 {
        let ne = n as f64 * f64::EPSILON;
        ne / (1.0 - ne)
    }

    /// The node count must put the aliasing term below one ulp: `nodes² ≥ 2η·ln(4/ε)`.
    fn assert_nodes_resolve(eta: f64) {
        let needed = (2.0 * eta * (4.0 / f64::EPSILON).ln()).sqrt();
        assert!(
            (NODES as f64) > needed.max(eta),
            "trapezoid rule with {NODES} nodes does not resolve eta={eta:e} (needs > {needed:e})"
        );
    }

    #[test]
    fn circle_log_marginal_equals_the_quadrature_of_its_integral_2933_f07() {
        for &period in &PERIODS {
            let k = TAU / period;
            for exponent in -12..=6 {
                let eta = 10.0_f64.powf(0.5 * exponent as f64);
                assert_nodes_resolve(eta);
                let kappa = eta * k * k;
                let (log_value, derivative) = circle_log_marginal(kappa.ln(), period);
                let (mass, mean_energy) = trapezoid_circle_moments(kappa, period, NODES);
                // Quadrature budget (see `NODES`) plus the closed form's own rounding: the
                // `ln` of `P`, the centered Bessel term and their sum, and `ln` of the mass.
                let value_band = gamma(NODES)
                    + (3.0 * eta + 1.0) * f64::EPSILON
                    + 4.0 * f64::EPSILON * (1.0 + period.ln().abs() + log_value.abs());
                assert!(
                    (log_value - mass.ln()).abs() <= value_band,
                    "P={period} eta={eta:e}: closed form log M={log_value:.17e}, quadrature \
                     {:.17e}, |diff|={:.3e} > band {value_band:.3e}",
                    mass.ln(),
                    (log_value - mass.ln()).abs(),
                );
                // `d log M / d log κ = −E[η(1 − cos k·u)]` under the matched von Mises law.
                let derivative_band = (2.0 * gamma(NODES) + 2.0 * f64::EPSILON) * mean_energy
                    + 2.0 * eta * f64::EPSILON
                    + 4.0 * f64::EPSILON * derivative.abs();
                assert!(
                    (derivative + mean_energy).abs() <= derivative_band,
                    "P={period} eta={eta:e}: closed-form d log M/d log kappa={derivative:.17e}, \
                     quadrature {:.17e}, |diff|={:.3e} > band {derivative_band:.3e}",
                    -mean_energy,
                    (derivative + mean_energy).abs(),
                );
            }
        }
    }

    #[test]
    fn circle_log_marginal_is_the_volume_at_zero_curvature_and_laplace_at_large_2933_f07() {
        for &period in &PERIODS {
            let k = TAU / period;
            // κ = 0 exactly: the coordinate's own volume, with no curvature score.
            let (log_value, derivative) = circle_log_marginal(f64::NEG_INFINITY, period);
            assert_eq!(log_value.to_bits(), period.ln().to_bits(), "P={period}: log M(0) != log P");
            assert_eq!(derivative.to_bits(), 0.0_f64.to_bits(), "P={period}: score at kappa = 0");
            for exponent in [-12, -8, -4, -1, 0, 1] {
                let eta = 10.0_f64.powi(exponent);
                let kappa = eta * k * k;
                let (log_value, _) = circle_log_marginal(kappa.ln(), period);
                // `1 ≤ I0(η) ≤ e^η` for η ≥ 0, so `log P − η ≤ log M ≤ log P` exactly.
                let rounding = 4.0 * f64::EPSILON * (1.0 + period.ln().abs());
                assert!(
                    log_value <= period.ln() + rounding && log_value >= period.ln() - eta - rounding,
                    "P={period} eta={eta:e}: log M={log_value:.17e} leaves [log P - eta, log P]"
                );
            }
            for exponent in [1, 3, 6, 12] {
                let eta = 10.0_f64.powi(exponent);
                let kappa = eta * k * k;
                let (log_value, derivative) = circle_log_marginal(kappa.ln(), period);
                let laplace = 0.5 * (TAU / kappa).ln();
                // `e^{−η}I0(η)√(2πη) = 1 + 1/(8η) + 9/(128η²) + …` with positive terms, so the
                // excess over the Laplace factor lies in `[0, (1 + 1/η)/(8η)]` for η ≥ 10.
                let excess = log_value - laplace;
                let rounding = 4.0 * f64::EPSILON * (1.0 + log_value.abs() + laplace.abs());
                let bound = (1.0 + 1.0 / eta) / (8.0 * eta);
                assert!(
                    excess >= -rounding && excess <= bound + rounding,
                    "P={period} eta={eta:e}: log M - log sqrt(2pi/kappa) = {excess:.3e} leaves \
                     [0, {bound:.3e}]"
                );
                // The derivative tends to the Laplace factor's −½ from below:
                // `η(I1/I0 − 1) = −½ − 1/(8η) − 1/(8η²) − 25/(128η³) − …`, inside
                // `(1 + 2/η)/(8η)` of −½ for η ≥ 25/16.
                let score_bound = (1.0 + 2.0 / eta) / (8.0 * eta);
                assert!(
                    (derivative + 0.5).abs() <= score_bound + 4.0 * f64::EPSILON,
                    "P={period} eta={eta:e}: d log M/d log kappa = {derivative:.17e}, not -1/2 \
                     within {score_bound:.3e}"
                );
            }
            // A curvature whose `κ` and `η` overflow still returns the Laplace factor in log
            // space, `log M = ½·log 2π − ½·log κ`.
            let log_kappa = 1.0e3;
            let (log_value, derivative) = circle_log_marginal(log_kappa, period);
            let laplace = 0.5 * (TAU.ln() - log_kappa);
            assert!(
                (log_value - laplace).abs() <= 4.0 * f64::EPSILON * (1.0 + log_kappa),
                "P={period}: log M at log kappa=1e3 is {log_value:.17e}, Laplace {laplace:.17e}"
            );
            assert_eq!(derivative.to_bits(), (-0.5_f64).to_bits(), "P={period}: overflow score");
        }
    }

    #[test]
    fn signed_circle_log_marginal_is_the_quadrature_at_every_curvature_sign_2933_f07() {
        for &period in &PERIODS {
            let k = TAU / period;
            let scale = 1.0 / (k * k);
            for exponent in -12..=5 {
                for sign in [-1.0_f64, 1.0] {
                    let eta = sign * 10.0_f64.powf(0.5 * exponent as f64);
                    assert_nodes_resolve(eta.abs());
                    let kappa = eta * k * k;
                    let (log_value, derivative) = circle_log_marginal_signed(kappa, period);
                    let (mass, mean_energy) = trapezoid_circle_moments(kappa, period, NODES);
                    // The positive-curvature budget of the unsigned pin; at `κ < 0` every
                    // summand's exponent is at most `2|η|`, so the same `(3|η| + 1)ε` holds.
                    let value_band = gamma(NODES)
                        + (3.0 * eta.abs() + 1.0) * f64::EPSILON
                        + 4.0 * f64::EPSILON * (1.0 + period.ln().abs() + log_value.abs());
                    assert!(
                        (log_value - mass.ln()).abs() <= value_band,
                        "P={period} eta={eta:e}: log M={log_value:.17e}, quadrature {:.17e}",
                        mass.ln(),
                    );
                    // `d log M / dκ = −E[(1 − cos k·u)]/k² = −E[η(1 − cos k·u)]/κ`.
                    let expected = -mean_energy / kappa;
                    let derivative_band =
                        (2.0 * gamma(NODES) + (2.0 * eta.abs() + 4.0) * f64::EPSILON)
                            * expected.abs().max(scale);
                    assert!(
                        (derivative - expected).abs() <= derivative_band,
                        "P={period} eta={eta:e}: d log M/d kappa={derivative:.17e}, quadrature \
                         {expected:.17e}, |diff|={:.3e} > {derivative_band:.3e}",
                        (derivative - expected).abs(),
                    );
                }
            }
            // κ = 0 is the period itself, with the score `−(P/2π)²` both sides reach.
            let (log_value, derivative) = circle_log_marginal_signed(0.0, period);
            assert_eq!(log_value.to_bits(), period.ln().to_bits(), "P={period}: log M(0)");
            assert!(
                (derivative + scale).abs() <= 2.0 * f64::EPSILON * scale,
                "P={period}: score at kappa = 0 is {derivative:e}, not {:e}",
                -scale
            );
            for kappa in [1.0e-300, -1.0e-300, f64::MIN_POSITIVE] {
                let (value, score) = circle_log_marginal_signed(kappa, period);
                assert!(
                    (value - period.ln()).abs() <= 4.0 * f64::EPSILON * (1.0 + period.ln().abs())
                        && (score + scale).abs() <= 4.0 * f64::EPSILON * scale,
                    "P={period} kappa={kappa:e}: ({value:e}, {score:e}) is not continuous at 0"
                );
            }
            // A positive curvature never holds more than the period.
            for exponent in -8..=8 {
                let kappa = 10.0_f64.powi(exponent);
                let (value, _) = circle_log_marginal_signed(kappa, period);
                assert!(value <= period.ln() + 4.0 * f64::EPSILON * (1.0 + period.ln().abs()));
            }
        }
    }

    #[test]
    fn ard_log_partition_periodic_factor_is_the_circle_marginal_2933_f07() {
        let supports = [
            CoordinatePriorSupport::Circle { period: 1.0 },
            CoordinatePriorSupport::Line,
            CoordinatePriorSupport::Circle { period: 1.0 },
        ];
        // The second periodic axis is priced at the quotient's half period (#2933 F25).
        let periods = [Some(1.0), None, Some(0.5)];
        for log_alpha_value in [-40.0, -6.0, 0.0, 6.0, 800.0] {
            let log_alpha = Array1::from_vec(vec![log_alpha_value; 3]);
            let alpha = log_alpha.mapv(f64::exp);
            let partition =
                SaeManifoldTerm::ard_log_partition(&supports, &periods, log_alpha.view(), alpha.view())
                    .expect("ARD log partition of a circle/line/circle atom");
            for (axis, period) in [(0, 1.0), (2, 0.5)] {
                let (log_volume, score) = circle_log_marginal(log_alpha_value, period);
                let paired = log_volume - 0.5 * TAU.ln();
                assert_eq!(
                    partition.per_factor[axis].to_bits(),
                    paired.to_bits(),
                    "log alpha={log_alpha_value}: axis {axis} partition {} != circle marginal {paired}",
                    partition.per_factor[axis],
                );
                assert_eq!(
                    partition.log_precision_gradient[axis].to_bits(),
                    score.to_bits(),
                    "log alpha={log_alpha_value}: axis {axis} score {} != {score}",
                    partition.log_precision_gradient[axis],
                );
            }
            // The line axis keeps its Gaussian normalizer.
            assert_eq!(partition.per_factor[1].to_bits(), (-0.5 * log_alpha_value).to_bits());
        }
    }
}

// #4077 — the Laplace mass of an interval coordinate pinned at an active bound.
//
// Included into `construction.rs` beside `construction_exact_hessian.rs`, whose pinned-slot
// projector (#3438) removes the slot from `A`.
//
// # The term
//
// At a slot `u` of row `i` the assembly pinned (`t ≤ lo` with `g > 0`, or `t ≥ hi` with `g < 0`),
// `A` carries the slot uncoupled at the metric's unit stiffness, so the pencil `(A, Φ)` prices it
// at `log 1 = 0`. The posterior there lives on the half-line inside the interval, whose Laplace
// mass is
//
//   J(μ, σ) = ∫₀^∞ exp(−μs − σs²/2) ds,    μ = |g_u|,  σ = A_raw[u, u],
//
// with `g_u` the raw inner gradient at the slot and `A_raw = B_raw + ΔC` the unprojected exact
// information. The criterion is a negative log evidence carrying `+½log|A|` with no
// per-coordinate `−½log 2π`: a free coordinate of curvature `σ` contributes `½log σ`, whose true
// negative log mass is `½log σ − ½log 2π`. In that convention the pinned slot contributes
// `−log J(μ, σ) + ½log 2π`, so `log|A|` gains
//
//   c_u = −2·log J(μ, σ) + log 2π.
//
// As `μ → 0⁺` the bound stops being strictly active, `J → ½√(2π/σ)`, and `c_u → log σ + 2·log 2`:
// half the Gaussian mass, exactly `log 2` (in the criterion's `½` units) above the free
// coordinate it becomes.
//
// # Its derivative
//
// `gam_math::gaussian_reciprocal::half_line_gaussian_log_jet` returns `[log J, κ₁, …]` with
// `κ₁ = −∂_μ log J` and `∂_σ log J = −½(κ₂ + κ₁²)`, so
//
//   dc_u = 2κ₁·sign(g_u)·dg_u + (κ₂ + κ₁²)·dσ.
//
// `dg_u` along `θ` is row `u` of `A_raw`, and along `ρ` it is the slot's entry of the implicit
// right-hand side `∂g/∂ρ` the outer gradient already forms. `dσ` is the unprojected
// `∂A_raw/∂(ρ, θ)` read on the one diagonal entry. The pinned coordinate itself never moves: its
// implicit response is zero.

/// Whether a θ-adjoint differentiates `A` itself, on which every slot the assembly pinned at an
/// active bound is a constant unit direction, or the unprojected `A_raw` (#4077).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinnedSlotDerivative {
    Projected,
    Raw,
}

/// One pinned slot's half-line operands, all read at the unprojected slot.
#[derive(Clone, Debug)]
pub(crate) struct PinnedHalfLineSlot {
    pub(crate) row: usize,
    /// Joint index `row_offsets[row] + local`.
    pub(crate) index: usize,
    /// The raw inner gradient `g_u`.
    pub(crate) gradient: f64,
    /// `A_raw[u, u]`.
    pub(crate) curvature: f64,
    /// `[log J, κ₁, κ₂, κ₃, κ₄]` at `(|g_u|, A_raw[u, u])`.
    pub(crate) jet: [f64; 5],
    /// Row `u` of `A_raw` on its row's coordinate block.
    pub(crate) row_curvature: Array1<f64>,
    /// Row `u` of `A_raw` on the border, as `(border index, value)`.
    pub(crate) border_curvature: Vec<(usize, f64)>,
}

impl PinnedHalfLineSlot {
    /// `c_u = −2·log J + log 2π`, in `log|A|` units.
    pub(crate) fn log_det_correction(&self) -> f64 {
        -2.0 * self.jet[0] + std::f64::consts::TAU.ln()
    }

    /// `∂c_u/∂g_u = 2κ₁·sign(g_u)`.
    pub(crate) fn gradient_coefficient(&self) -> f64 {
        2.0 * self.jet[1] * self.gradient.signum()
    }

    /// `∂c_u/∂σ = κ₂ + κ₁²`.
    pub(crate) fn curvature_coefficient(&self) -> f64 {
        self.jet[2] + self.jet[1] * self.jet[1]
    }
}

/// `Σ_u w_u·e_u e_uᵀ` over pinned coordinate slots: the weight whose contraction against
/// `∂A_raw` reads each slot's own curvature derivative.
struct PinnedSlotWeight {
    /// `(joint index, weight)`, sorted by index.
    entries: Vec<(usize, f64)>,
    /// The coordinate count and border width of the joint layout the weight was built on.
    total_t: usize,
    border_dim: usize,
    /// The one stored value of the zero border block, read through a zero-stride view.
    zero: [f64; 1],
}

impl PinnedSlotWeight {
    fn new(slots: &[PinnedHalfLineSlot], cache: &ArrowFactorCache) -> Self {
        let mut entries: Vec<(usize, f64)> = slots
            .iter()
            .map(|slot| (slot.index, slot.curvature_coefficient()))
            .collect();
        entries.sort_by_key(|&(index, _)| index);
        Self {
            entries,
            total_t: cache.delta_t_len(),
            border_dim: cache.k,
            zero: [0.0],
        }
    }
}

impl JointWeight for PinnedSlotWeight {
    fn entry(&self, row: usize, column: usize) -> f64 {
        if row != column {
            return 0.0;
        }
        self.entries
            .binary_search_by_key(&row, |&(index, _)| index)
            .map_or(0.0, |position| self.entries[position].1)
    }

    fn row_block(&self, base: usize, q: usize) -> Array2<f64> {
        let mut block = Array2::<f64>::zeros((q, q));
        let start = self.entries.partition_point(|&(index, _)| index < base);
        for &(index, weight) in &self.entries[start..] {
            if index >= base + q {
                break;
            }
            block[[index - base, index - base]] = weight;
        }
        block
    }

    fn border_block(&self, total_t: usize) -> Option<ArrayView2<'_, f64>> {
        // Every entry is a coordinate slot, so the border block is zero.
        let shape = ndarray::ShapeBuilder::strides(
            (self.border_dim, self.border_dim),
            (0, 0),
        );
        (total_t == self.total_t)
            .then(|| ArrayView2::from_shape(shape, &self.zero[..]).ok())
            .flatten()
    }

    fn dense(&self) -> Option<&Array2<f64>> {
        None
    }

    fn coordinate_diagonal_only(&self) -> bool {
        true
    }
}

impl SaeManifoldTerm {
    /// Every slot the last assembly pinned at an active bound, with its raw gradient, its
    /// unprojected exact curvature row and its half-line jet, in row order. Empty when nothing
    /// is pinned.
    ///
    /// The row's jets and residual are the ones the assembly and `ΔC` read: `B_raw`'s slot row
    /// is the Gauss--Newton `J_uᵀ M J` with the ARD majorizer on its diagonal, and `ΔC`'s is
    /// [`Self::exact_hessian_minus_b_row_raw`], the owner the row assembly projects.
    pub(crate) fn pinned_half_line_slots(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<PinnedHalfLineSlot>, String> {
        if self.last_pinned_bound_slots.is_empty() {
            return Ok(Vec::new());
        }
        let pinned = self.pinned_bound_slots_by_row(&cache.row_dims)?;
        let operands = self.exact_hessian_delta_operands(rho)?;
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut slots = Vec::new();
        for (row, locals) in pinned.iter().enumerate() {
            if locals.is_empty() {
                continue;
            }
            let vars = self.row_vars_for_row_dim(row, cache.row_dims[row])?;
            self.assignment.try_assignments_row_into(
                row,
                assignments.as_slice_mut().ok_or_else(|| {
                    "pinned_half_line_slots: assignment scratch is not contiguous".to_string()
                })?,
            )?;
            let jets =
                self.row_jets_for_logdet(row, vars, assignments.view(), &second_jets, &border)?;
            let w_row = self.row_loss_weights.as_deref().map_or(1.0, |w| w[row]);
            let error_metric =
                self.patchd_row_error_metric(row, w_row, target, &assignments, operands.whitens);
            let delta = self.exact_hessian_minus_b_row_raw(
                &operands,
                row,
                &jets,
                &error_metric,
                &assignments,
                w_row,
                border.len(),
            );
            let q = jets.vars.len();
            for &local in locals {
                let SaeLocalRowVar::Coord { atom, axis } = jets.vars[local] else {
                    return Err(format!(
                        "pinned_half_line_slots: pinned slot (row {row}, slot {local}) is not a \
                         coordinate: {:?}",
                        jets.vars[local]
                    ));
                };
                let first = jets.first(local);
                // `M J_u`, the metric the likelihood's Gauss--Newton block reads.
                let metric_first: Vec<f64> = match self.row_metric.as_ref() {
                    Some(metric) if operands.whitens => {
                        metric.apply_metric_row(row, ndarray::aview1(first))
                    }
                    _ => first.to_vec(),
                };
                let mut gradient = sae_dot(first, &error_metric);
                let mut row_curvature = Array1::<f64>::zeros(q);
                for other in 0..q {
                    row_curvature[other] =
                        sae_dot(&metric_first, jets.first(other)) + delta.tt[[local, other]];
                }
                if operands.ard_live[atom] {
                    let alpha = operands.ard_precisions[atom][axis];
                    let t_value = self.assignment.coords[atom].row(row)[axis];
                    let prior =
                        ArdAxisPrior::eval(alpha, t_value, operands.ard_axis_periods[atom][axis]);
                    gradient += w_row * prior.grad;
                    row_curvature[local] += w_row * prior.psd_majorizer_hess();
                }
                let border_curvature = border
                    .iter()
                    .enumerate()
                    .map(|(position, channel)| {
                        (
                            channel.index,
                            sae_dot(&metric_first, jets.beta(position))
                                + delta.tbeta[[local, position]],
                        )
                    })
                    .collect();
                let curvature = row_curvature[local];
                let jet = gam_math::gaussian_reciprocal::half_line_gaussian_log_jet(
                    gradient.abs(),
                    curvature,
                )
                .ok_or_else(|| {
                    format!(
                        "pinned_half_line_slots: row {row} slot {local} has no half-line mass at \
                         gradient {gradient:.6e} and curvature {curvature:.6e}"
                    )
                })?;
                slots.push(PinnedHalfLineSlot {
                    row,
                    index: cache.row_offsets[row] + local,
                    gradient,
                    curvature,
                    jet,
                    row_curvature,
                    border_curvature,
                });
            }
        }
        Ok(slots)
    }

    /// `Σ_u c_u` over the pinned slots, in `log|A|` units: the half-line mass every exact-`A`
    /// value route adds to its log-determinant. Zero when nothing is pinned.
    pub(crate) fn pinned_half_line_log_det_correction(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<f64, String> {
        Ok(self
            .pinned_half_line_slots(rho, target, cache)?
            .iter()
            .map(PinnedHalfLineSlot::log_det_correction)
            .sum())
    }

    /// The half-line mass's log-determinant channels at fixed `θ`, in the conventions of every
    /// exact-`A` channel: the `ρ` trace carries the criterion's leading `½` and the θ-adjoint is
    /// in full `log|A|` units. The trace holds the curvature leg `½(κ₂ + κ₁²)·∂σ/∂ρ`; the
    /// gradient leg `κ₁·sign(g)·∂g_u/∂ρ` reads the outer gradient's own implicit right-hand side
    /// and is added where that is formed ([`Self::pinned_half_line_gradient_legs`]).
    ///
    /// `None` when nothing is pinned.
    pub(crate) fn pinned_half_line_channels(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<Option<(Array1<f64>, SaeArrowVector)>, String> {
        let slots = self.pinned_half_line_slots(rho, target, cache)?;
        if slots.is_empty() {
            return Ok(None);
        }
        let weight = PinnedSlotWeight::new(&slots, cache);
        let mut trace = Array1::<f64>::zeros(rho.flat_coordinates().len());
        let mut curvature_traces = ContractingPenaltyDerivatives::new(&weight);
        self.raw_penalty_curvature_operators_into(rho, cache, &mut curvature_traces)?;
        self.exact_stationarity_penalty_derivative_delta_into(rho, cache, &mut curvature_traces)?;
        for (flat, contraction) in curvature_traces.contractions {
            trace[flat] += 0.5 * contraction;
        }
        // #2231 — a crosscoder block weight scales its target columns, which `A_raw[u, u]`
        // reads through the residual. `A_raw` is affine in the target and
        // `∂Z̃_ℓ/∂log λ_ℓ = ½Z̃_ℓ`, so the curvature at the target with block `ℓ` scaled by
        // `1.5` differs from the base curvature by exactly `∂σ/∂log λ_ℓ`.
        if let Some((p_x, block_dims)) = self.crosscoder_pricing_spans.as_ref() {
            let range = rho.block_flat_range();
            if range.len() != block_dims.len() {
                return Err(format!(
                    "pinned_half_line_channels: rho carries {} block coordinates for {} \
                     crosscoder blocks",
                    range.len(),
                    block_dims.len()
                ));
            }
            let mut shifted = target.to_owned();
            let mut start = *p_x;
            for (coord, &width) in range.zip(block_dims.iter()) {
                shifted.assign(&target);
                shifted
                    .slice_mut(s![.., start..start + width])
                    .mapv_inplace(|value| 1.5 * value);
                let moved = self.pinned_half_line_slots(rho, shifted.view(), cache)?;
                for (base, moved) in slots.iter().zip(moved.iter()) {
                    trace[coord] +=
                        0.5 * base.curvature_coefficient() * (moved.curvature - base.curvature);
                }
                start += width;
            }
        }
        let mut theta = self.logdet_theta_adjoint_dense_on_slots(
            rho,
            cache,
            &weight,
            true,
            true,
            Some(target),
            PinnedSlotDerivative::Raw,
        )?;
        for slot in &slots {
            let coefficient = slot.gradient_coefficient();
            let base = cache.row_offsets[slot.row];
            for (other, &value) in slot.row_curvature.iter().enumerate() {
                theta.t[base + other] += coefficient * value;
            }
            for &(index, value) in &slot.border_curvature {
                theta.beta[index] += coefficient * value;
            }
        }
        Ok(Some((trace, theta)))
    }

    /// Zero the pinned slots of an implicit right-hand side `∂g/∂ρ`, in place.
    ///
    /// The inner stationarity the IFT differentiates is the PROJECTED gradient, and at a slot
    /// pinned at an active bound that is identically zero for every nearby `(ρ, θ)`: the
    /// retraction holds the coordinate on the bound. So the slot's entry of `∂g/∂ρ` is zero,
    /// as an embedded sphere's normal component is, and with `A` carrying the slot as the
    /// uncoupled unit direction the mode response `−A⁻¹∂g/∂ρ` leaves it exactly still. The
    /// raw entry is what the half-line mass reads ([`Self::pinned_half_line_gradient_legs`]),
    /// so a caller reads that first.
    pub(crate) fn project_pinned_ift_rhs(&self, cache: &ArrowFactorCache, rhs: &mut SaeArrowVector) {
        for &(row, local) in &self.last_pinned_bound_slots {
            rhs.t[cache.row_offsets[row] + local] = 0.0;
        }
    }

    /// `(joint index, ½·∂c_u/∂g_u)` per pinned slot: the criterion-unit coefficient the outer
    /// gradient multiplies each coordinate's implicit right-hand side entry `∂g_u/∂ρ` by.
    pub(crate) fn pinned_half_line_gradient_legs(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<(usize, f64)>, String> {
        Ok(self
            .pinned_half_line_slots(rho, target, cache)?
            .iter()
            .map(|slot| (slot.index, 0.5 * slot.gradient_coefficient()))
            .collect())
    }
}

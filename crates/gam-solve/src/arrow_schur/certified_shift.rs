//! The proximal-ridge rungs an arrow system's declared bounds certify (#2627).
//!
//! `solve_with_lm_escalation_inner` and the proximal correction both add one ridge
//! `μ` to `H_tt^(i) + ridge_t·I` and to the reduced border `ridge_β·I`, and re-solve
//! at a geometrically larger `μ` after a rejection. A count of rungs is a budget,
//! not a reason to stop: it gives up below the operator's own scale on a large
//! system and runs far past it on a small one. The reasons to stop are structural.
//!
//! * Once the shifted system provably passes every conditioning guard its
//!   factorization applies, a larger shift cannot cure a refusal
//!   ([`ArrowShiftCertificate::certifies_factorable`]).
//! * Once the damped model promises a decrease below the objective's representable
//!   resolution, a larger shift cannot produce one
//!   ([`ArrowShiftCertificate::damped_curvature_lower_bound`]).
//!
//! # Bounds read once per system
//!
//! * Row `i`: Gershgorin edges `g_i ≤ λ_min` and `G_i ≥ λ_max` of `H_tt^(i)`, with
//!   radii `Σ_{b≠a} max(|h_ab|, |h_ba|)`, which cover whichever triangle a
//!   factorization reads.
//! * `s_i ≥ ‖H_tβ^(i)‖₂`, from [`ArrowSchurSystem::cross_block_row_norm_bounds`].
//! * `N_β`, the largest guaranteed row sum of the shared block's majorant. It bounds
//!   `‖H_ββ‖₂` and `‖|H_ββ|‖₂`, since the majorant dominates `|H_ββ|` entrywise, and
//!   the majorant's accumulation depth `k_β` prices the apply's rounding.
//!
//! Every combination of bounds is carried in the direction it bounds. One rounded
//! operation moves its result by at most `u` relative, and [`round_down`] /
//! [`round_up`] charge it, so a lower bound stays below the value it bounds.
//!
//! # A factorable rung
//!
//! At `ρ_t = ridge_t + μ` and `ρ_β = ridge_β + μ`, row `i`'s damped block has
//! `λ_min ≥ lo_i = g_i + ρ_t` and `λ_max ≤ hi_i = G_i + ρ_t`. A computed Cholesky
//! factor is the exact factor of `Â = A + ΔA` with `|ΔA| ≤ γ_{d+1}·|L̂||L̂ᵀ|` (Higham,
//! *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Theorem 10.3), so
//! `‖ΔA‖₂ ≤ c_d·‖Â‖₂` with `c_d = d·γ_{d+1}` and `‖ΔA‖₂ ≤ e_i = c_d/(1 − c_d)·hi_i`.
//! Every Cholesky pivot of `Â` lies in `[λ_min(Â), λ_max(Â)]`, so the factor's
//! diagonal-ratio κ estimate is at most `κ_i = (hi_i + e_i)/(lo_i − e_i)` and its
//! smallest pivot at least `lo_i − e_i`. The row passes the production safe-inversion
//! guard when
//!
//! ```text
//! lo_i − e_i ≥ safe_spd_pivot_min(diag_scale_i)   and   hi_i + e_i ≤ safe_spd_kappa_max(d_i)·(lo_i − e_i).
//! ```
//!
//! The reduced border from those factors, `S = H_ββ + ρ_β·I − Σ_i H_βt^(i) Â_i⁻¹ H_tβ^(i)`,
//! has `‖H_βt^(i) Â_i⁻¹ H_tβ^(i)‖₂ ≤ q_i = s_i²/(lo_i − e_i)`, so
//!
//! ```text
//! λ_min(S) ≥ σ_lo = ρ_β − N_β − Σ_i q_i,        λ_max(S) ≤ σ_hi = N_β + ρ_β.
//! ```
//!
//! The border as computed differs from `S` by charged bands:
//!
//! * the substitutions. `(L̂ + ΔL)ŷ = b` with `|ΔL| ≤ γ_d·|L̂|` (Higham, Theorem 8.5)
//!   moves a solved column by at most `η_i = γ_d·√d·√κ_i / (1 − γ_d·√d·√κ_i)` relative,
//!   and at most two substitutions per column give `ω_i = (1 + η_i)² − 1`. A row's
//!   term, formed as `ŶᵀŶ` or as `H_βt Ẑ`, then moves by at most `ω_i·d_i·q_i`;
//! * the accumulation of `|H_ββ| + ρ_β·I + Σ_i |H_βt^(i)||Ẑ_i|`, of depth
//!   `2·k_x + max_i d_i + n + 2` per entry and 2-norm at most
//!   `N_β + ρ_β + Σ_i (1 + ω_i)·d_i·q_i`, plus the shared apply's own `γ_{k_β}·N_β`. Here
//!   `k_x` is the cross block's declared apply depth: a matrix-free apply of a row term
//!   runs the forward apply, then the transpose, the sum over the `n` rows, the shared
//!   apply and the ridge, and a dense assembly's `d_i`-term products summed over the rows
//!   fall within the same count;
//! * on a route that factors the dense border, that factor's backward error,
//!   `c_K/(1 − c_K)` times `σ_hi` plus the bands above.
//!
//! With `e_S` their sum, every route keeps positive curvature when `σ_lo − e_S > 0`. A
//! route that factors the dense border also refuses a κ estimate past
//! `safe_spd_kappa_max(K)`, which `σ_hi + e_S ≤ safe_spd_kappa_max(K)·(σ_lo − e_S)`
//! rules out.
//!
//! No constant is chosen here: the coefficients are `u`, the guards' own thresholds,
//! and bounds the system declares. A loose bound certifies a later rung, which only
//! adds refused rungs before the typed refusal; it never changes which step a rung
//! accepts.

use super::*;
use gam_linalg::roundoff::accumulation_growth;

/// A lower bound on `z` from `fl(z)`, the rounded result of one operation on
/// floating-point operands: `z ≥ fl(z) − γ_1·|fl(z)|`, and forming that band and the
/// difference round twice more, each by at most `u` relative, which `γ_3` covers.
pub(crate) fn round_down(computed: f64) -> f64 {
    computed - accumulation_growth(3) * computed.abs()
}

/// The upper counterpart of [`round_down`].
pub(crate) fn round_up(computed: f64) -> f64 {
    computed + accumulation_growth(3) * computed.abs()
}

/// `c/(1 − c)` for the backward-error coefficient `c = n·γ_{n+1}` of an `n×n`
/// Cholesky factor, or `+∞` when `c ≥ 1` leaves no bound.
pub(crate) fn cholesky_backward_error_ratio(dim: usize) -> f64 {
    let coefficient = round_up(dim as f64 * accumulation_growth(dim.saturating_add(1)));
    if !(coefficient < 1.0) {
        return f64::INFINITY;
    }
    round_up(coefficient / round_down(1.0 - coefficient))
}

/// The relative band `ω = (1 + η)² − 1` of at most two substitutions on the Cholesky
/// factor of a `d×d` block with condition number at most `kappa`, where
/// `η = γ_d·√d·√κ / (1 − γ_d·√d·√κ)` and `√κ` bounds the factor's own condition number;
/// `None` when `γ_d·√d·√κ ≥ 1` leaves no bound.
pub(crate) fn substitution_band(dim: usize, kappa: f64) -> Option<f64> {
    let solve = round_up(
        round_up(accumulation_growth(dim) * (dim as f64).sqrt()) * round_up(kappa.sqrt()),
    );
    if !(solve < 1.0) {
        return None;
    }
    let eta = round_up(solve / round_down(1.0 - solve));
    Some(round_up(round_up(2.0 * eta) + round_up(eta * eta)))
}

/// `N_β ≥ ‖H_ββ‖₂` and `N_β ≥ ‖|H_ββ|‖₂`, the largest guaranteed row sum of the shared
/// block's majorant, and the majorant's accumulation depth `k_β`, which prices the shared
/// apply's rounding (#2627).
pub(crate) fn shared_block_norm_bound(sys: &ArrowSchurSystem) -> Result<(f64, usize), String> {
    let ones = vec![1.0_f64; sys.k];
    let mut row_sums = vec![0.0_f64; sys.k];
    let depth = sys.shared_block_abs_majorant_matvec(&ones, &mut row_sums);
    let mut norm = 0.0_f64;
    for &sum in &row_sums {
        let bound = guaranteed_norm_upper_bound(sum, depth);
        if !bound.is_finite() {
            return Err(format!(
                "the shared block's majorant row sum {sum:e} at depth {depth} carries no finite \
                 bound"
            ));
        }
        norm = norm.max(bound);
    }
    Ok((norm, depth))
}

/// The accumulation depth of one reduced-border entry, applied or assembled (#2627): the
/// cross block's forward and transpose applies (`cross_apply_depth` each), the sum over
/// `rows` rows, the shared apply and the ridge. A dense assembly's `d_i`-term products
/// summed over the rows fall within the same count.
pub(crate) fn border_accumulation_depth(cross_apply_depth: usize, widest_row: usize, rows: usize) -> usize {
    cross_apply_depth
        .saturating_mul(2)
        .saturating_add(widest_row)
        .saturating_add(rows)
        .saturating_add(2)
}

/// One row's bounds, in the units of `H_tt^(i)` without any ridge.
#[derive(Debug, Clone, Copy)]
struct RowShiftBounds {
    dim: usize,
    lower_edge: f64,
    upper_edge: f64,
    pivot_floor: f64,
    cross_norm_squared: f64,
}

/// The damped quantities of one row at `ρ_t`: the factor's certified smallest
/// pivot `lo − e` and the relative substitution band `ω`, or `None` when this row
/// does not pass the safe-inversion guard at `ρ_t`.
#[derive(Debug, Clone, Copy)]
struct DampedRow {
    pivot_lower: f64,
    substitution_band: f64,
}

impl RowShiftBounds {
    fn damped(&self, ridge_t: f64) -> Option<DampedRow> {
        let lower = round_down(self.lower_edge + ridge_t);
        let upper = round_up(self.upper_edge + ridge_t);
        let band = round_up(cholesky_backward_error_ratio(self.dim) * lower.abs().max(upper.abs()));
        let pivot_lower = round_down(lower - band);
        let eigen_upper = round_up(upper + band);
        if !(pivot_lower > 0.0 && pivot_lower >= self.pivot_floor) {
            return None;
        }
        if !(eigen_upper <= round_down(safe_spd_kappa_max(self.dim) * pivot_lower)) {
            return None;
        }
        let kappa = round_up(eigen_upper / pivot_lower);
        let substitution_band = substitution_band(self.dim, kappa)?;
        Some(DampedRow {
            pivot_lower,
            substitution_band,
        })
    }
}

/// The bounds an arrow system declares, read once, and the rung predicates built
/// from them. See the module documentation for the derivation.
#[derive(Debug, Clone)]
pub(crate) struct ArrowShiftCertificate {
    rows: Vec<RowShiftBounds>,
    ridge_t: f64,
    ridge_beta: f64,
    border_dim: usize,
    border_norm: f64,
    border_apply_depth: usize,
    accumulation_depth: usize,
    gradient_squared: f64,
}

impl ArrowShiftCertificate {
    /// Read the bounds of `sys` at the base ridges. Refuses a system with a
    /// non-finite entry or bound, which no shift can certify, and a matrix-free
    /// cross block installed without its declared bounds.
    pub(crate) fn from_system(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<Self, String> {
        let (cross_norms, cross_apply_depth) = sys.cross_block_row_norm_bounds()?;
        let mut rows = Vec::with_capacity(sys.rows.len());
        let mut widest_row = 0usize;
        for (index, (row, &cross_norm)) in sys.rows.iter().zip(cross_norms.iter()).enumerate() {
            let dim = row.htt.nrows();
            widest_row = widest_row.max(dim);
            let mut lower_edge = f64::INFINITY;
            let mut upper_edge = f64::NEG_INFINITY;
            for a in 0..dim {
                let mut radius = 0.0_f64;
                for b in 0..dim {
                    if b != a {
                        radius += row.htt[[a, b]].abs().max(row.htt[[b, a]].abs());
                    }
                }
                let radius = guaranteed_norm_upper_bound(radius, dim);
                let diagonal = row.htt[[a, a]];
                let lower = round_down(diagonal - radius);
                let upper = round_up(diagonal + radius);
                if !(lower.is_finite() && upper.is_finite()) {
                    return Err(format!(
                        "row {index} H_tt carries a non-finite entry or Gershgorin radius, \
                         so no proximal shift can be certified"
                    ));
                }
                lower_edge = lower_edge.min(lower);
                upper_edge = upper_edge.max(upper);
            }
            let cross_norm_squared = round_up(cross_norm * cross_norm);
            if !cross_norm_squared.is_finite() {
                return Err(format!(
                    "row {index} H_tβ norm bound {cross_norm:e} is not finite, so no proximal \
                     shift can be certified"
                ));
            }
            rows.push(RowShiftBounds {
                dim,
                lower_edge,
                upper_edge,
                pivot_floor: safe_spd_pivot_min(row_block_diag_scale(row, dim)),
                cross_norm_squared,
            });
        }
        let (border_norm, border_apply_depth) = shared_block_norm_bound(sys)?;
        let mut gradient_sum = 0.0_f64;
        let mut gradient_terms = 0usize;
        for value in sys.rows.iter().flat_map(|row| row.gt.iter()).chain(sys.gb.iter()) {
            gradient_sum += value * value;
            gradient_terms += 1;
        }
        let gradient_squared = guaranteed_norm_upper_bound(gradient_sum, gradient_terms);
        if !gradient_squared.is_finite() {
            return Err(format!(
                "the gradient's squared norm {gradient_sum:e} carries no finite bound"
            ));
        }
        let accumulation_depth = border_accumulation_depth(cross_apply_depth, widest_row, sys.rows.len());
        Ok(Self {
            rows,
            ridge_t,
            ridge_beta,
            border_dim: sys.k,
            border_norm,
            border_apply_depth,
            accumulation_depth,
            gradient_squared,
        })
    }

    /// Whether every factorization guard provably passes with the proximal ridge
    /// `proximal_ridge` added to both base ridges. `dense_border_factor` states that
    /// the route factors the dense reduced border and so also applies its κ guard.
    pub(crate) fn certifies_factorable(&self, proximal_ridge: f64, dense_border_factor: bool) -> bool {
        let ridge_t = self.ridge_t + proximal_ridge;
        let ridge_beta = self.ridge_beta + proximal_ridge;
        let mut coupled = 0.0_f64;
        let mut substitution = 0.0_f64;
        let mut formed = 0.0_f64;
        for row in &self.rows {
            if row.dim == 0 {
                continue;
            }
            let Some(damped) = row.damped(ridge_t) else {
                return false;
            };
            let coupling = round_up(row.cross_norm_squared / damped.pivot_lower);
            let spread = round_up(row.dim as f64 * coupling);
            coupled += coupling;
            substitution += round_up(damped.substitution_band * spread);
            formed += round_up(round_up(1.0 + damped.substitution_band) * spread);
        }
        if self.border_dim == 0 {
            return true;
        }
        let terms = self.rows.len().saturating_add(1);
        let coupled = guaranteed_norm_upper_bound(coupled, terms);
        let substitution = guaranteed_norm_upper_bound(substitution, terms);
        let formed = guaranteed_norm_upper_bound(formed, terms);
        let lower = round_down(round_down(ridge_beta - self.border_norm) - coupled);
        let upper = round_up(self.border_norm + ridge_beta);
        let accumulated = round_up(
            accumulation_growth(self.accumulation_depth)
                * round_up(round_up(self.border_norm + ridge_beta) + formed),
        );
        let applied = round_up(accumulation_growth(self.border_apply_depth) * self.border_norm);
        let mut band = round_up(round_up(accumulated + applied) + substitution);
        if dense_border_factor {
            let factor = round_up(cholesky_backward_error_ratio(self.border_dim) * round_up(upper + band));
            band = round_up(band + factor);
        }
        let margin = round_down(lower - band);
        if !(margin > 0.0) {
            return false;
        }
        !dense_border_factor
            || round_up(upper + band) <= round_down(safe_spd_kappa_max(self.border_dim) * margin)
    }

    /// A guaranteed lower bound on `λ_min` of the joint Hessian with the proximal
    /// ridge added to both base ridges, or `None` when these bounds certify no
    /// positive curvature there.
    ///
    /// With `m_i = g_i + ρ_t`, `m = min_i m_i`, `c = ρ_β − N_β` and `T = Σ_i s_i²/m_i`,
    /// `H + shift − ν·I` is positive semidefinite when `ν < m` and
    /// `c − ν − Σ_i s_i²/(m_i − ν) ≥ 0` (the Schur complement of the rows). Since
    /// `m_i − ν ≥ m_i·(1 − ν/m)`, the left side is at least `c − ν − T·m/(m − ν)`, whose
    /// root below `m` is `ν = 2m(c − T)/((c + m) + √((c − m)² + 4mT))`, positive exactly
    /// when `c > T`. It increases in `c` and `m` and decreases in `T`, so evaluating it on
    /// directed bounds keeps it a lower bound.
    pub(crate) fn damped_curvature_lower_bound(&self, proximal_ridge: f64) -> Option<f64> {
        let ridge_t = self.ridge_t + proximal_ridge;
        let ridge_beta = self.ridge_beta + proximal_ridge;
        let mut row_floor = f64::INFINITY;
        let mut coupled = 0.0_f64;
        for row in &self.rows {
            if row.dim == 0 {
                continue;
            }
            let lower = round_down(row.lower_edge + ridge_t);
            if !(lower > 0.0) {
                return None;
            }
            row_floor = row_floor.min(lower);
            coupled += round_up(row.cross_norm_squared / lower);
        }
        let border_floor = round_down(ridge_beta - self.border_norm);
        if self.border_dim == 0 {
            return row_floor.is_finite().then_some(row_floor);
        }
        if !row_floor.is_finite() {
            return (border_floor > 0.0).then_some(border_floor);
        }
        let coupled = guaranteed_norm_upper_bound(coupled, self.rows.len().saturating_add(1));
        let excess = round_down(border_floor - coupled);
        if !(excess > 0.0) {
            return None;
        }
        let numerator = round_down(2.0 * row_floor * excess);
        let gap = round_up((border_floor - row_floor).abs());
        let discriminant = round_up(round_up(gap * gap) + round_up(4.0 * row_floor * coupled));
        let denominator =
            round_up(round_up(border_floor + row_floor) + round_up(discriminant.sqrt()));
        let floor = round_down(numerator / denominator);
        (floor > 0.0).then_some(floor)
    }

    /// Whether the damped Newton model at this rung promises less decrease than half
    /// the float spacing at the incumbent value `objective`, inside which a trial value
    /// rounds to the incumbent.
    ///
    /// The model's largest decrease is `½·gᵀ(H + shift)⁻¹g ≤ ½‖g‖²/ν`, with `ν` from
    /// [`Self::damped_curvature_lower_bound`], so the predicate is `‖g‖²/ν < spacing`.
    /// `ν` grows with the ridge, so once it holds it holds at every later rung. It reads
    /// certified curvature, never the ridge alone: `‖g‖²/μ` understates the decrease
    /// wherever `H` has a negative eigenvalue.
    pub(crate) fn promises_unrepresentable_decrease(&self, proximal_ridge: f64, objective: f64) -> bool {
        let Some(curvature) = self.damped_curvature_lower_bound(proximal_ridge) else {
            return false;
        };
        let magnitude = objective.abs();
        let spacing = magnitude.next_up() - magnitude;
        round_up(self.gradient_squared / curvature) < spacing
    }
}

impl ArrowSchurSystem {
    /// Guaranteed upper bounds `s_i ≥ ‖H_tβ^(i)‖₂` of the cross block the solve applies,
    /// and that apply's accumulation depth (#2627).
    ///
    /// The bounds are the installed matrix-free operator's declaration plus the Frobenius
    /// bound of a dense slab the apply adds, exactly as `sys_htbeta_apply_row` adds it. A
    /// dense slab's forward is an inner product over `K` columns per latent coordinate and
    /// its transpose one over `d` coordinates, and adding either into an existing entry is
    /// one more operation. Refuses an installed operator that carries no declaration, or
    /// one declared for a different row count.
    pub(crate) fn cross_block_row_norm_bounds(&self) -> Result<(Vec<f64>, usize), String> {
        let declared = match (self.htbeta_matvec.as_ref(), self.htbeta_declaration.as_ref()) {
            (None, _) => None,
            (Some(_), Some(declaration)) if declaration.row_norm_bounds.len() == self.rows.len() => {
                Some(declaration)
            }
            (Some(_), Some(declaration)) => {
                return Err(format!(
                    "the matrix-free H_tβ operator declares {} row norm bounds for {} rows",
                    declaration.row_norm_bounds.len(),
                    self.rows.len()
                ));
            }
            (Some(_), None) => {
                return Err(
                    "a matrix-free H_tβ operator is installed without its declaration; \
                     install it through ArrowSchurSystem::set_row_htbeta_operator"
                        .to_string(),
                );
            }
        };
        let dense_slab_applies = self.htbeta_dense_supplement || self.htbeta_matvec.is_none();
        let widest_latent = self.rows.iter().map(|row| row.htt.nrows()).max().unwrap_or(0);
        let slab_depth = if dense_slab_applies {
            self.k.max(widest_latent)
        } else {
            0
        };
        let apply_depth = declared
            .map_or(0, |declaration| declaration.apply_depth)
            .max(slab_depth)
            .saturating_add(1);
        let bounds = self
            .rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let operator = declared.map_or(0.0, |declaration| declaration.row_norm_bounds[index]);
                let slab = if dense_slab_applies && row.htbeta.dim() == (row.htt.nrows(), self.k) {
                    frobenius_norm_upper_bound(row.htbeta.iter().copied())
                } else {
                    0.0
                };
                guaranteed_norm_upper_bound(operator + slab, 1)
            })
            .collect();
        Ok((bounds, apply_depth))
    }
}

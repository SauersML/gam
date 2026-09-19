//! gam#2922: the Bernoulli marginal-slope Jeffreys information as the expected
//! Fisher information `I(β) = Σ_i w_i·∇p_i∇p_iᵀ/(p_i(1 − p_i))`, the object
//! `family_trait.rs` requires of a non-canonical Bernoulli likelihood.
//!
//! A binary row log-likelihood `ℓ = y·ln p + (1 − y)·ln(1 − p)` has
//! `∇ℓ = (y − p)/(p(1 − p))·∇p`, so for either outcome
//! `∇p∇pᵀ/(p(1 − p)) = κ(ℓ)·∇ℓ∇ℓᵀ` with `κ(ℓ) = 1/expm1(−ℓ)`: `p/(1 − p)` at
//! `y = 1` and `(1 − p)/p` at `y = 0`. The information and its coefficient
//! derivatives are therefore read off the row kernels the likelihood already
//! evaluates: the row negative log-likelihood, its primary gradient and Hessian,
//! and its contracted third and fourth derivatives. `κ' = κ + κ²`,
//! `κ'' = κ + 3κ² + 2κ³` and `κ''' = κ + 7κ² + 12κ³ + 6κ⁴`.
//!
//! Every product is formed through `s = κ·∇ℓ`, attaching each power of `κ` to a
//! gradient factor, never through a bare `κ²`. Near saturation `κ` grows like
//! `1/Φ(−m)` while `∇ℓ` shrinks like `φ(m)`, and `s` stays of the size of the margin
//! `m`. A bare `κ` is left only where it multiplies Hessian or third-derivative
//! products, which are of the size of `Φ(−m)` themselves.
//!
//! A row's primaries are linear in the coefficients: the marginal and slope
//! predictors are design-row dot products and the flexible primaries are the
//! deviation coefficients themselves (`pullback_primary_vector_add_into`). So the
//! row Jacobian `J_i` does not move with `β`, and every coefficient derivative of
//! `Σ_i J_iᵀ M_i J_i` is the pullback of the primary-space derivative of `M_i`.

use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::contract_fourth_full;
use super::hessian_paths::*;
use super::*;
use crate::custom_family::JeffreysInformationMotion;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

/// `κ(ℓ) = 1/expm1(−ℓ)` at a row's per-unit-weight log-likelihood.
///
/// `None` when `expm1(−ℓ)` is not positive or `κ` overflows. The row log-likelihood
/// is `ln Φ` at the signed margin, formed from the complementary tail
/// (`gam_math::probability::normal_logcdf_derivatives`), so `expm1(−ℓ)` rounds to
/// zero only once `Φ(−m)` has underflowed. The expected weight
/// `φ(m)²/(Φ(m)Φ(−m))` is then itself below the smallest normal double, and the
/// row carries no resolved information. On the misclassified side `expm1(−ℓ)`
/// overflows instead, `κ = 0`, and the zero weight is again what rounding says.
fn expected_information_odds(log_likelihood: f64) -> Option<f64> {
    let complement_odds = (-log_likelihood).exp_m1();
    let odds = complement_odds.recip();
    (complement_odds > 0.0 && odds.is_finite()).then_some(odds)
}

/// `target += coefficient·½(a·bᵀ + b·aᵀ)`.
fn add_symmetric_outer(
    target: &mut Array2<f64>,
    coefficient: f64,
    first: &Array1<f64>,
    second: &Array1<f64>,
) {
    let half = 0.5 * coefficient;
    for (row, (&first_row, &second_row)) in first.iter().zip(second.iter()).enumerate() {
        for (column, (&first_column, &second_column)) in
            first.iter().zip(second.iter()).enumerate()
        {
            target[[row, column]] += half * (first_row * second_column + second_row * first_column);
        }
    }
}

/// One row's expected information in primary space, per unit weight: the
/// log-likelihood gradient `g` and Hessian `h`, and `s = κ·g`.
struct ExpectedInformationRow {
    weight: f64,
    odds: f64,
    gradient: Array1<f64>,
    scaled: Array1<f64>,
    hessian: Array2<f64>,
}

impl ExpectedInformationRow {
    /// From the weighted row negative log-likelihood channels. `None` when the row
    /// carries no resolved information: a zero weight, or an unresolved `κ`.
    fn from_negative_log_likelihood(
        weight: f64,
        negative_log_likelihood: f64,
        gradient: ArrayView1<'_, f64>,
        hessian: ArrayView2<'_, f64>,
    ) -> Option<Self> {
        if !(weight > 0.0) {
            return None;
        }
        let odds = expected_information_odds(-negative_log_likelihood / weight)?;
        let gradient = gradient.mapv(|value| -value / weight);
        let scaled = gradient.mapv(|value| odds * value);
        let hessian = hessian.mapv(|value| -value / weight);
        Some(Self {
            weight,
            odds,
            gradient,
            scaled,
            hessian,
        })
    }

    fn dimension(&self) -> usize {
        self.gradient.len()
    }

    /// A third-derivative matrix of the weighted row NLL, per unit weight.
    fn per_unit_weight(&self, mut weighted_nll_matrix: Array2<f64>) -> Array2<f64> {
        let weight = self.weight;
        weighted_nll_matrix.mapv_inplace(|value| -value / weight);
        weighted_nll_matrix
    }

    /// `w·κ·ggᵀ = w·sym(s gᵀ)`.
    fn information(&self) -> Array2<f64> {
        let dimension = self.dimension();
        let mut out = Array2::zeros((dimension, dimension));
        add_symmetric_outer(&mut out, self.weight, &self.scaled, &self.gradient);
        out
    }

    /// `D_x(w·κ·ggᵀ) = w·[κ'·(gᵀx)·ggᵀ + κ·(h x gᵀ + g (h x)ᵀ)]`.
    fn first_directional(&self, direction: &Array1<f64>) -> Array2<f64> {
        self.first_directional_parts(self.gradient.dot(direction), &self.hessian.dot(direction))
    }

    /// [`Self::first_directional`] from the slot's derivatives `Dℓ` and `D∇ℓ`, which a
    /// hyperparameter the row reads directly supplies itself.
    fn first_directional_parts(&self, along: f64, moved: &Array1<f64>) -> Array2<f64> {
        let dimension = self.dimension();
        let mut out = Array2::zeros((dimension, dimension));
        add_symmetric_outer(&mut out, along, &self.scaled, &self.gradient);
        add_symmetric_outer(&mut out, along, &self.scaled, &self.scaled);
        add_symmetric_outer(&mut out, 2.0, moved, &self.scaled);
        out.mapv_inplace(|value| value * self.weight);
        out
    }

    /// `D_x D_y(w·κ·ggᵀ)` with `third_along_second = T[·, ·, y]` per unit weight:
    ///
    /// ```text
    /// κ''·aₓa_y·ggᵀ + κ'·xᵀhy·ggᵀ + κ'·a_y·2sym(hx gᵀ) + κ'·aₓ·2sym(hy gᵀ)
    ///   + κ·2sym(T[x, y] gᵀ) + κ·2sym(hy (hx)ᵀ)
    /// ```
    fn second_directional(
        &self,
        first: &Array1<f64>,
        second: &Array1<f64>,
        third_along_second: &Array2<f64>,
    ) -> Array2<f64> {
        let second_moved = self.hessian.dot(second);
        self.second_directional_parts(
            self.gradient.dot(first),
            self.scaled.dot(first),
            &self.hessian.dot(first),
            self.gradient.dot(second),
            self.scaled.dot(second),
            &second_moved,
            first.dot(&second_moved),
            &third_along_second.dot(first),
        )
    }

    /// [`Self::second_directional`] from the slots' derivatives: `Dℓ`, `κ·Dℓ` and `D∇ℓ` of each
    /// slot, and the mixed `D_xD_yℓ` and `D_xD_y∇ℓ`. A primary direction passes `s·x` for `κ·Dℓ`,
    /// the product its formula has always formed.
    fn second_directional_parts(
        &self,
        first_along: f64,
        first_scaled: f64,
        first_moved: &Array1<f64>,
        second_along: f64,
        second_scaled: f64,
        second_moved: &Array1<f64>,
        curvature: f64,
        third_moved: &Array1<f64>,
    ) -> Array2<f64> {
        let dimension = self.dimension();
        let mut out = Array2::zeros((dimension, dimension));
        add_symmetric_outer(
            &mut out,
            first_along * second_along + curvature,
            &self.scaled,
            &self.gradient,
        );
        add_symmetric_outer(
            &mut out,
            3.0 * first_along * second_along + 2.0 * first_scaled * second_along + curvature,
            &self.scaled,
            &self.scaled,
        );
        add_symmetric_outer(
            &mut out,
            2.0 * (second_along + second_scaled),
            first_moved,
            &self.scaled,
        );
        add_symmetric_outer(
            &mut out,
            2.0 * (first_along + first_scaled),
            second_moved,
            &self.scaled,
        );
        add_symmetric_outer(&mut out, 2.0, third_moved, &self.scaled);
        add_symmetric_outer(&mut out, 2.0 * self.odds, second_moved, first_moved);
        out.mapv_inplace(|value| value * self.weight);
        out
    }

    /// `∇²_x tr(W_i·w κ ggᵀ)` for the row's projected trace weight `W_i`, with
    /// `third_along_scaled_weight = T[·, ·, W_i s]` per unit weight:
    ///
    /// ```text
    /// κ''·c·ggᵀ + κ'·c·h + 2κ'·2sym(h W_i g gᵀ) + 2κ·T[W_i g] + 2κ·h W_i h,  c = gᵀW_i g
    /// ```
    fn contracted_trace_hessian(
        &self,
        weight_in_primary: &Array2<f64>,
        third_along_scaled_weight: &Array2<f64>,
    ) -> Array2<f64> {
        let dimension = self.dimension();
        let weighted_gradient = weight_in_primary.dot(&self.gradient);
        let weighted_scaled = weight_in_primary.dot(&self.scaled);
        let contraction = self.gradient.dot(&weighted_gradient);
        let scaled_contraction = self.scaled.dot(&weighted_gradient);
        let moved = self.hessian.dot(&weighted_gradient);
        let scaled_moved = self.hessian.dot(&weighted_scaled);
        let mut out = Array2::zeros((dimension, dimension));
        add_symmetric_outer(&mut out, contraction, &self.scaled, &self.gradient);
        add_symmetric_outer(
            &mut out,
            3.0 * contraction + 2.0 * scaled_contraction,
            &self.scaled,
            &self.scaled,
        );
        out.scaled_add((1.0 + self.odds) * scaled_contraction, &self.hessian);
        add_symmetric_outer(&mut out, 4.0, &moved, &self.scaled);
        add_symmetric_outer(&mut out, 4.0, &scaled_moved, &self.scaled);
        out.scaled_add(2.0, third_along_scaled_weight);
        let curvature_product = self.hessian.dot(weight_in_primary).dot(&self.hessian);
        out.scaled_add(2.0 * self.odds, &curvature_product);
        out.mapv_inplace(|value| value * self.weight);
        out
    }

    /// `D_x D_y D_{e_k}(w·κ ggᵀ)` along primary axis `k`, with `third_along_first =
    /// T[·, ·, x]`, `third_along_second = T[·, ·, y]` and `fourth = Q[·, ·, x, y]`,
    /// all per unit weight (see the module documentation for the κ derivatives):
    ///
    /// ```text
    /// κ'''·aₓa_y a_z·G + κ''·(ℓ_xy a_z + ℓ_xz a_y + ℓ_yz aₓ)·G
    ///   + κ''·(aₓa_y G_z + aₓa_z G_y + a_y a_z Gₓ) + κ'·ℓ_xyz·G
    ///   + κ'·(ℓ_xy G_z + ℓ_xz G_y + ℓ_yz Gₓ) + κ'·(aₓ G_yz + a_y G_xz + a_z G_xy) + κ·G_xyz
    /// ```
    fn third_directional_axis(
        &self,
        first: &Array1<f64>,
        second: &Array1<f64>,
        third_along_first: &Array2<f64>,
        third_along_second: &Array2<f64>,
        fourth: &Array2<f64>,
        axis: usize,
    ) -> Array2<f64> {
        let first_moved = self.hessian.dot(first);
        let second_moved = self.hessian.dot(second);
        let pair_moved = third_along_second.dot(first);
        self.third_directional_axis_parts(
            &RowLikelihoodDerivative {
                along: self.gradient.dot(first),
                scaled: self.scaled.dot(first),
                moved: &first_moved,
                curvature: third_along_first,
            },
            &RowLikelihoodDerivative {
                along: self.gradient.dot(second),
                scaled: self.scaled.dot(second),
                moved: &second_moved,
                curvature: third_along_second,
            },
            &RowLikelihoodDerivative {
                along: first.dot(&second_moved),
                scaled: self.odds * first.dot(&second_moved),
                moved: &pair_moved,
                curvature: fourth,
            },
            axis,
        )
    }

    /// [`Self::third_directional_axis`] from the slots' derivatives: `x` and `y` each carry
    /// `Dℓ`, `D∇ℓ` and `D∇²ℓ`, and `pair` carries the mixed `D_xD_yℓ`, `D_xD_y∇ℓ` and
    /// `D_xD_y∇²ℓ`. The formula differentiates along any derivation that commutes with the
    /// primary axis `k`, so a hyperparameter the row reads directly is a slot too.
    fn third_directional_axis_parts(
        &self,
        x: &RowLikelihoodDerivative<'_>,
        y: &RowLikelihoodDerivative<'_>,
        pair: &RowLikelihoodDerivative<'_>,
        axis: usize,
    ) -> Array2<f64> {
        let dimension = self.dimension();
        let odds = self.odds;
        let (a_x, a_y, a_z) = (x.along, y.along, self.gradient[axis]);
        let (b_x, b_y, b_z) = (x.scaled, y.scaled, self.scaled[axis]);
        let g_x = x.moved;
        let g_y = y.moved;
        let g_z = self.hessian.column(axis).to_owned();
        let l_xy = pair.along;
        let l_xz = g_x[axis];
        let l_yz = g_y[axis];
        let g_xy = pair.moved;
        let g_xz = x.curvature.column(axis).to_owned();
        let g_yz = y.curvature.column(axis).to_owned();
        let g_xyz = pair.curvature.column(axis).to_owned();
        let l_xyz = g_xy[axis];
        let mut out = Array2::zeros((dimension, dimension));
        let s = &self.scaled;
        let g = &self.gradient;
        add_symmetric_outer(&mut out, a_x * a_y * a_z, s, g);
        add_symmetric_outer(
            &mut out,
            7.0 * a_x * a_y * a_z + 12.0 * b_x * a_y * a_z + 6.0 * b_x * b_y * a_z,
            s,
            s,
        );
        let mixed = l_xy * a_z + l_xz * a_y + l_yz * a_x;
        let mixed_scaled = l_xy * b_z + l_xz * b_y + l_yz * b_x;
        add_symmetric_outer(&mut out, mixed, s, g);
        add_symmetric_outer(&mut out, 3.0 * mixed + 2.0 * mixed_scaled, s, s);
        for (a_p, b_p, a_q, b_q, g_r) in [
            (a_x, b_x, a_y, b_y, &g_z),
            (a_x, b_x, a_z, b_z, g_y),
            (a_y, b_y, a_z, b_z, g_x),
        ] {
            add_symmetric_outer(&mut out, 2.0 * (a_p * a_q + 3.0 * b_p * a_q + 2.0 * b_p * b_q), g_r, s);
        }
        add_symmetric_outer(&mut out, l_xyz, s, g);
        add_symmetric_outer(&mut out, l_xyz, s, s);
        for (l_pq, g_r) in [(l_xy, &g_z), (l_xz, g_y), (l_yz, g_x)] {
            add_symmetric_outer(&mut out, 2.0 * (1.0 + odds) * l_pq, g_r, s);
        }
        for (a_p, b_p, g_qr, g_q, g_r) in [
            (a_x, b_x, &g_yz, g_y, &g_z),
            (a_y, b_y, &g_xz, g_x, &g_z),
            (a_z, b_z, g_xy, g_x, g_y),
        ] {
            add_symmetric_outer(&mut out, 2.0 * (a_p + b_p), g_qr, s);
            add_symmetric_outer(&mut out, 2.0 * (b_p + odds * b_p), g_q, g_r);
        }
        add_symmetric_outer(&mut out, 2.0, &g_xyz, s);
        add_symmetric_outer(&mut out, 2.0 * odds, g_xy, &g_z);
        add_symmetric_outer(&mut out, 2.0 * odds, &g_xz, g_y);
        add_symmetric_outer(&mut out, 2.0 * odds, &g_yz, g_x);
        out.mapv_inplace(|value| value * self.weight);
        out
    }
}

/// The coefficient axes that move primary `primary_axis` of row `i`, with the Jacobian
/// entries `J_i[primary_axis, a]`: the marginal or slope design row for those two
/// primaries, and the identity for a deviation coefficient.
fn primary_axis_coefficients(
    slices: &BlockSlices,
    primary: &PrimarySlices,
    primary_axis: usize,
    marginal_row: &Array2<f64>,
    slope_row: &Array2<f64>,
) -> Vec<(usize, f64)> {
    if primary_axis == primary.q {
        return slices
            .marginal
            .clone()
            .zip(marginal_row.row(0).iter().copied())
            .collect();
    }
    if primary_axis == primary.slope {
        return slices
            .slope
            .clone()
            .zip(slope_row.row(0).iter().copied())
            .collect();
    }
    [
        (primary.h.as_ref(), slices.h.as_ref()),
        (primary.w.as_ref(), slices.w.as_ref()),
    ]
    .into_iter()
    .find_map(|(primary_range, block_range)| match (primary_range, block_range) {
        (Some(primary_range), Some(block_range)) if primary_range.contains(&primary_axis) => {
            Some((block_range.start + (primary_axis - primary_range.start), 1.0))
        }
        _ => None,
    })
    .into_iter()
    .collect()
}

/// A ψ axis's design row at row `i` and the primary motion it causes at fixed `β`:
/// `δ_i = e_q·(∂_ψx_i·β_block)`.
fn expected_information_psi_motion(
    axis: &PsiAxisSpec,
    block_states: &[ParameterBlockState],
    primary: &PrimarySlices,
    row: usize,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    expected_information_design_motion(&axis.psi_map, axis, block_states, primary, row)
}

/// Row `i` of a design derivative map on `axis`'s block and the primary motion it causes at
/// fixed `β`: `e_q·(m_i·β_block)` on the primary `q` that block feeds.
fn expected_information_design_motion(
    map: &gam_custom_family::PsiDesignMap,
    axis: &PsiAxisSpec,
    block_states: &[ParameterBlockState],
    primary: &PrimarySlices,
    row: usize,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let design_row = map
        .row_vector(row)
        .map_err(|error| format!("BMS expected information psi map row {row}: {error}"))?;
    let mut motion = Array1::<f64>::zeros(primary.total);
    motion[axis.idx_primary] = design_row.dot(&block_states[axis.block_idx].beta);
    Ok((design_row, motion))
}

/// The coefficient range of the design block a ψ axis moves.
fn psi_block_range(slices: &BlockSlices, axis: &PsiAxisSpec) -> std::ops::Range<usize> {
    if axis.block_idx == 0 {
        slices.marginal.clone()
    } else {
        slices.slope.clone()
    }
}

/// `axes[start + a] += design_row[a]·moved`: a primary motion `e_q·design_row[a]` scattered
/// onto the coefficient axes of the design block that moves it.
fn add_design_row_scaled(
    axes: &mut [Array2<f64>],
    start: usize,
    design_row: &Array1<f64>,
    moved: &Array2<f64>,
) {
    for (local, &scale) in design_row.iter().enumerate() {
        if scale != 0.0 {
            axes[start + local].scaled_add(scale, moved);
        }
    }
}

/// The two design axes of a ψ pair, with the second design map `∂²X/∂ψ_i∂ψ_j` when both
/// axes move one block.
pub(super) struct ExpectedInformationPsiPair {
    first: PsiAxisSpec,
    second: PsiAxisSpec,
    second_design: Option<gam_custom_family::PsiDesignMap>,
}

/// One derivative of a row's log-likelihood per unit weight with its primary gradient and
/// Hessian: along a primary direction, along a hyperparameter the row reads directly, or mixed
/// between two of those. `scaled` is `κ·along`; a primary direction passes `s·x`, the product
/// its formula has always formed.
struct RowLikelihoodDerivative<'a> {
    along: f64,
    scaled: f64,
    moved: &'a Array1<f64>,
    curvature: &'a Array2<f64>,
}

/// A hyperparameter the expected information moves with.
enum ExpectedInformationAxis {
    /// A derivative of the marginal or slope design.
    Design(PsiAxisSpec),
    /// `t = log σ` of the Gaussian frailty.
    LogFrailtyScale,
}

/// Which frailty-scale derivative of the expected information one rigid pass assembles.
#[derive(Clone, Copy)]
enum LogFrailtyOrder<'a> {
    /// `∂_t I|_β`.
    Value,
    /// `∂²_t I|_β`.
    Second,
    /// `∂_t∂_ψ I|_β` along a design axis.
    Design(&'a PsiAxisSpec),
}

/// Which frailty-scale derivative of the expected information one rigid pass assembles on every
/// coefficient axis.
#[derive(Clone, Copy)]
enum LogFrailtyAxesOrder<'a> {
    /// `{∂_t D_β I[e_a]}`.
    First,
    /// `{∂²_t D_β I[e_a]}`.
    Second,
    /// `{∂_t∂_ψ D_β I[e_a]}` along a design axis.
    Design(&'a PsiAxisSpec),
    /// `{∂_t D²_β I[direction, e_a]}`.
    Directional(&'a Array1<f64>),
}

/// Which coefficient derivative of the expected information one pass assembles.
#[derive(Clone, Copy)]
enum ExpectedInformationOrder<'a> {
    Value,
    First(&'a Array1<f64>),
    Second(&'a Array1<f64>, &'a Array1<f64>),
    /// `∇²_β tr(W·I(β))` for a symmetric coefficient-space weight `W`.
    ContractedTraceHessian(&'a Array2<f64>),
}

impl BernoulliMarginalSlopeFamily {
    /// A fit with the residual repair block prices its Jeffreys term from the observed joint
    /// Hessian, the only information its row kernel supplies, so it publishes no
    /// expected-information motion.
    pub(super) fn residual_jeffreys_motion_declined<T>(&self) -> Option<JeffreysInformationMotion<T>> {
        self.residual_active()
            .then(|| JeffreysInformationMotion::Unpublished {
                reason: "a residual-repair fit prices its Jeffreys term from the observed joint Hessian"
                    .to_string(),
            })
    }

    /// The coefficient count the specs declare against the family's own blocks, which index
    /// the expected information's rows and columns (gam#2922).
    pub(super) fn expected_information_specs_match(
        &self,
        specs: &[ParameterBlockSpec],
        what: &str,
    ) -> Result<(), String> {
        let declared = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        let total = block_slices(self).total;
        if declared != total {
            return Err(format!(
                "BMS {what}: the specs declare {declared} coefficients and the family's blocks {total}"
            ));
        }
        Ok(())
    }

    /// The expected Fisher information of the coefficient system (gam#2922).
    pub(super) fn expected_jeffreys_information(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array2<f64>, String> {
        self.expected_information_pullback(block_states, ExpectedInformationOrder::Value)
    }

    /// `D_β I[direction]` of [`Self::expected_jeffreys_information`].
    pub(super) fn expected_jeffreys_information_directional(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.expected_information_pullback(block_states, ExpectedInformationOrder::First(direction))
    }

    /// `D²_β I[first, second]` of [`Self::expected_jeffreys_information`].
    pub(super) fn expected_jeffreys_information_second_directional(
        &self,
        block_states: &[ParameterBlockState],
        first: &Array1<f64>,
        second: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.expected_information_pullback(
            block_states,
            ExpectedInformationOrder::Second(first, second),
        )
    }

    /// `∇²_β tr(W·I(β))`, the contracted trace Hessian the Jeffreys completion
    /// reads: one contracted row third per row, along `W_i·s`.
    pub(super) fn expected_jeffreys_information_contracted_trace_hessian(
        &self,
        block_states: &[ParameterBlockState],
        weight: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let total = block_slices(self).total;
        if weight.dim() != (total, total) {
            return Err(format!(
                "BMS expected information contracted trace: weight shape {:?} != ({total}, {total})",
                weight.dim()
            ));
        }
        let mut symmetric = weight + &weight.t();
        symmetric.mapv_inplace(|value| 0.5 * value);
        self.expected_information_pullback(
            block_states,
            ExpectedInformationOrder::ContractedTraceHessian(&symmetric),
        )
    }

    /// `{D_β I[e_a]}` for every coefficient axis `a`, or `{D²_β I[direction, e_a]}` with a
    /// direction: one exact cache and one row state per row, and one pullback per primary
    /// scattered through `J_i[k, a]`. The per-axis hooks would rebuild the exact cache on every
    /// axis (gam#2892). The second derivative reads one contracted row third per row.
    pub(super) fn expected_jeffreys_information_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        direction: Option<&Array1<f64>>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        if let Some(direction) = direction
            && direction.len() != total
        {
            return Err(format!(
                "BMS expected information drift expected a direction of length {total}, got {}",
                direction.len()
            ));
        }
        let flexible = self.effective_flex_active(block_states)?;
        match direction {
            Some(_) => self.prewarm_expected_information_third(
                block_states,
                &cache,
                flexible,
                "expected Jeffreys information second drift",
            )?,
            None if flexible => cache
                .row_primary_hessians
                .reject_device_cpu_recompute("expected Jeffreys information drift")?,
            None => {}
        }
        let empty = || vec![Array2::<f64>::zeros((total, total)); total];
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut axes = empty();
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                let mut marginal_row = Array2::<f64>::zeros((1, slices.marginal.len()));
                let mut slope_row = Array2::<f64>::zeros((1, slices.slope.len()));
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let along = direction
                        .map(|direction| -> Result<_, String> {
                            let x = self.row_primary_direction_from_flat(row, slices, primary, direction)?;
                            let third =
                                self.expected_information_row_third(row, block_states, &cache, flexible, &state, &x)?;
                            Ok((x, third))
                        })
                        .transpose()?;
                    self.marginal_design
                        .row_chunk_into(row..row + 1, marginal_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    self.slope_design
                        .row_chunk_into(row..row + 1, slope_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    for primary_axis in 0..primary.total {
                        let mut unit = Array1::<f64>::zeros(primary.total);
                        unit[primary_axis] = 1.0;
                        let basis = match along.as_ref() {
                            Some((x, third)) => state.second_directional(&unit, x, third),
                            None => state.first_directional(&unit),
                        };
                        let mut pulled = BernoulliBlockHessianAccumulator::new(slices);
                        pulled.add_pullback(self, row, slices, primary, &basis);
                        let pulled = pulled.to_dense(slices);
                        for (coefficient_axis, scale) in primary_axis_coefficients(
                            slices,
                            primary,
                            primary_axis,
                            &marginal_row,
                            &slope_row,
                        ) {
                            if scale != 0.0 {
                                axes[coefficient_axis].scaled_add(scale, &pulled);
                            }
                        }
                    }
                }
                Ok(axes)
            },
            |mut left, right| -> Result<_, String> {
                for (left_axis, right_axis) in left.iter_mut().zip(right.iter()) {
                    *left_axis += right_axis;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(empty);
        Ok(accumulated)
    }

    /// `{D³_β I[first, second, e_a]}` for every coefficient axis `a`: one contracted
    /// row fourth per row along `(first, second)`, and one pullback per primary.
    pub(super) fn expected_jeffreys_information_third_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        first: &Array1<f64>,
        second: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        let flexible = self.effective_flex_active(block_states)?;
        self.prewarm_expected_information_fourth(
            block_states,
            &cache,
            flexible,
            "expected Jeffreys information third derivative",
        )?;
        let empty = || vec![Array2::<f64>::zeros((total, total)); total];
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut axes = empty();
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                let mut marginal_row = Array2::<f64>::zeros((1, slices.marginal.len()));
                let mut slope_row = Array2::<f64>::zeros((1, slices.slope.len()));
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let x = self.row_primary_direction_from_flat(row, slices, primary, first)?;
                    let y = self.row_primary_direction_from_flat(row, slices, primary, second)?;
                    let third_x =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &x)?;
                    let third_y =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &y)?;
                    let fourth = self.expected_information_row_fourth(
                        row,
                        block_states,
                        &cache,
                        flexible,
                        &state,
                        &x,
                        &y,
                    )?;
                    self.marginal_design
                        .row_chunk_into(row..row + 1, marginal_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    self.slope_design
                        .row_chunk_into(row..row + 1, slope_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    for axis in 0..primary.total {
                        let basis = state.third_directional_axis(
                            &x, &y, &third_x, &third_y, &fourth, axis,
                        );
                        let mut pulled = BernoulliBlockHessianAccumulator::new(slices);
                        pulled.add_pullback(self, row, slices, primary, &basis);
                        let pulled = pulled.to_dense(slices);
                        for (coefficient_axis, scale) in primary_axis_coefficients(
                            slices,
                            primary,
                            axis,
                            &marginal_row,
                            &slope_row,
                        ) {
                            if scale != 0.0 {
                                axes[coefficient_axis].scaled_add(scale, &pulled);
                            }
                        }
                    }
                }
                Ok(axes)
            },
            |mut left, right| -> Result<_, String> {
                for (left_axis, right_axis) in left.iter_mut().zip(right.iter()) {
                    *left_axis += right_axis;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(empty);
        Ok(accumulated)
    }

    /// `∂_ψ I|_β` along one design hyperparameter. The ψ-moved design row moves the
    /// primary it feeds by `δ_i = e_q·(∂_ψx_i·β)`, so the row information moves by
    /// `D M_i[δ_i]`, and the design row itself moves the pullback by the rank-one
    /// cross `(∂_ψJ_i)ᵀM_iJ_i + J_iᵀM_i(∂_ψJ_i)`: the observed ψ row pass with the
    /// expected information in place of the row Hessian.
    pub(super) fn expected_jeffreys_information_psi_derivative(
        &self,
        block_states: &[ParameterBlockState],
        axis: &PsiAxisSpec,
    ) -> Result<Array2<f64>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let flexible = self.effective_flex_active(block_states)?;
        if flexible {
            cache
                .row_primary_hessians
                .reject_device_cpu_recompute("expected Jeffreys information psi derivative")?;
        }
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut accumulator = BernoulliBlockHessianAccumulator::new(slices);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let (psi_row, motion) = expected_information_psi_motion(axis, block_states, primary, row)?;
                    accumulator.add_pullback(self, row, slices, primary, &state.first_directional(&motion));
                    let information = state.information();
                    accumulator.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        axis.block_idx,
                        &psi_row,
                        information.row(axis.idx_primary),
                    )?;
                }
                Ok(accumulator)
            },
            |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            },
        )?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(accumulated.to_dense(slices))
    }

    /// `{∂_ψ D_β I[e_a]|_β}` for every coefficient axis `a`:
    /// `D²M_i[x_a, δ_i] + D M_i[(∂_ψJ_i)e_a]` pulled back, plus the rank-one cross of
    /// the moved design row with `D M_i[x_a]`. One contracted row third per row, along
    /// `δ_i`.
    pub(super) fn expected_jeffreys_information_psi_directional_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        axis: &PsiAxisSpec,
    ) -> Result<Vec<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        let flexible = self.effective_flex_active(block_states)?;
        if flexible {
            cache
                .row_primary_hessians
                .reject_device_cpu_recompute("expected Jeffreys information psi drift")?;
            // The contracted third reads the degree-15 cell bundle; build it before the
            // parallel fold so no worker races its lazy construction.
            self.prewarm_flex_cell_bundle(block_states, &cache, 15)?;
        }
        let psi_block = psi_block_range(slices, axis);
        let empty = || vec![Array2::<f64>::zeros((total, total)); total];
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut axes = empty();
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                let mut marginal_row = Array2::<f64>::zeros((1, slices.marginal.len()));
                let mut slope_row = Array2::<f64>::zeros((1, slices.slope.len()));
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let (psi_row, motion) = expected_information_psi_motion(axis, block_states, primary, row)?;
                    let third_along_motion =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &motion)?;
                    self.marginal_design
                        .row_chunk_into(row..row + 1, marginal_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    self.slope_design
                        .row_chunk_into(row..row + 1, slope_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    let mut moved_design_axis = None;
                    for primary_axis in 0..primary.total {
                        let mut unit = Array1::<f64>::zeros(primary.total);
                        unit[primary_axis] = 1.0;
                        let first = state.first_directional(&unit);
                        let mut pulled = BernoulliBlockHessianAccumulator::new(slices);
                        pulled.add_pullback(
                            self,
                            row,
                            slices,
                            primary,
                            &state.second_directional(&unit, &motion, &third_along_motion),
                        );
                        pulled.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            axis.block_idx,
                            &psi_row,
                            first.row(axis.idx_primary),
                        )?;
                        let pulled = pulled.to_dense(slices);
                        for (coefficient_axis, scale) in primary_axis_coefficients(
                            slices,
                            primary,
                            primary_axis,
                            &marginal_row,
                            &slope_row,
                        ) {
                            if scale != 0.0 {
                                axes[coefficient_axis].scaled_add(scale, &pulled);
                            }
                        }
                        if primary_axis == axis.idx_primary {
                            let mut first_pulled = BernoulliBlockHessianAccumulator::new(slices);
                            first_pulled.add_pullback(self, row, slices, primary, &first);
                            moved_design_axis = Some(first_pulled.to_dense(slices));
                        }
                    }
                    // `(∂_ψJ_i)e_a` moves primary `q` by `∂_ψx_i[a]` on the ψ block's axes.
                    if let Some(moved) = moved_design_axis {
                        add_design_row_scaled(&mut axes, psi_block.start, &psi_row, &moved);
                    }
                }
                Ok(axes)
            },
            |mut left, right| -> Result<_, String> {
                for (left_axis, right_axis) in left.iter_mut().zip(right.iter()) {
                    *left_axis += right_axis;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(empty);
        Ok(accumulated)
    }

    /// `∂_{ψ_i}∂_{ψ_j} I|_β` along two design hyperparameters. The primary motions are
    /// `δ_i = e_q·(∂_ψx_i·β)` and `δ_ij = e_q·(∂²_ψx_i·β)`, and the moved design rows
    /// `J_i = e_q r_iᵀ` and `J_ij = e_q r_ijᵀ` enter through `×(A, X) = AᵀXJ + JᵀXA`:
    ///
    /// ```text
    /// Jᵀ(D²M[δ_i, δ_j] + D M[δ_ij])J + ×(J_i, D M[δ_j]) + ×(J_j, D M[δ_i]) + ×(J_ij, M)
    ///   + M_{q_iq_j}·(r_i r_jᵀ + r_j r_iᵀ)
    /// ```
    ///
    /// the observed ψψ row pass with the expected information in place of the row Hessian.
    pub(super) fn expected_jeffreys_information_psi_second_derivative(
        &self,
        block_states: &[ParameterBlockState],
        pair: &ExpectedInformationPsiPair,
    ) -> Result<Array2<f64>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let flexible = self.effective_flex_active(block_states)?;
        self.prewarm_expected_information_third(
            block_states,
            &cache,
            flexible,
            "expected Jeffreys information psi pair",
        )?;
        let (axis_i, axis_j) = (&pair.first, &pair.second);
        let (q_i, q_j) = (axis_i.idx_primary, axis_j.idx_primary);
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut accumulator = BernoulliBlockHessianAccumulator::new(slices);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let (row_i, motion_i) = expected_information_psi_motion(axis_i, block_states, primary, row)?;
                    let (row_j, motion_j) = expected_information_psi_motion(axis_j, block_states, primary, row)?;
                    let third_j =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &motion_j)?;
                    let information = state.information();
                    let mut moved = state.second_directional(&motion_i, &motion_j, &third_j);
                    if let Some(map) = pair.second_design.as_ref() {
                        let (row_ij, motion_ij) =
                            expected_information_design_motion(map, axis_i, block_states, primary, row)?;
                        moved += &state.first_directional(&motion_ij);
                        accumulator.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            axis_i.block_idx,
                            &row_ij,
                            information.row(q_i),
                        )?;
                    }
                    accumulator.add_pullback(self, row, slices, primary, &moved);
                    accumulator.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        axis_i.block_idx,
                        &row_i,
                        state.first_directional(&motion_j).row(q_i),
                    )?;
                    accumulator.add_rank1_psi_cross(
                        self,
                        row,
                        slices,
                        primary,
                        axis_j.block_idx,
                        &row_j,
                        state.first_directional(&motion_i).row(q_j),
                    )?;
                    accumulator.add_psi_psi_outer(
                        axis_i.block_idx,
                        &row_i,
                        axis_j.block_idx,
                        &row_j,
                        information[[q_i, q_j]],
                    );
                }
                Ok(accumulator)
            },
            |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            },
        )?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(accumulated.to_dense(slices))
    }

    /// `{D_β I_{ψ_iψ_j}[e_a]}` for every coefficient axis `a`, the coefficient derivative of
    /// [`Self::expected_jeffreys_information_psi_second_derivative`]. Along primary `k` a row
    /// contributes `D³M[δ_i, δ_j, e_k] + D²M[δ_ij, e_k]` pulled back, `×(J_i, D²M[δ_j, e_k])`,
    /// `×(J_j, D²M[δ_i, e_k])`, `×(J_ij, D M[e_k])` and `D M[e_k]_{q_iq_j}·(r_i r_jᵀ + r_j r_iᵀ)`,
    /// scattered through `J[k, a]`. The motions move with `β` as well, `D_β δ_i[e_a] = e_q·r_i[a]`,
    /// which adds `r_i[a]·(D²M[e_{q_i}, δ_j] + ×(J_j, D M[e_{q_i}]))`,
    /// `r_j[a]·(D²M[e_{q_j}, δ_i] + ×(J_i, D M[e_{q_j}]))` and `r_ij[a]·D M[e_{q_i}]` pulled back
    /// on the moved blocks. One contracted row fourth per row, along `(δ_i, δ_j)`.
    pub(super) fn expected_jeffreys_information_psi_second_derivative_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        pair: &ExpectedInformationPsiPair,
    ) -> Result<Vec<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        let flexible = self.effective_flex_active(block_states)?;
        self.prewarm_expected_information_fourth(
            block_states,
            &cache,
            flexible,
            "expected Jeffreys information psi pair drift",
        )?;
        let (axis_i, axis_j) = (&pair.first, &pair.second);
        let (q_i, q_j) = (axis_i.idx_primary, axis_j.idx_primary);
        let (block_i, block_j) = (psi_block_range(slices, axis_i), psi_block_range(slices, axis_j));
        let empty = || vec![Array2::<f64>::zeros((total, total)); total];
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut axes = empty();
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                let mut marginal_row = Array2::<f64>::zeros((1, slices.marginal.len()));
                let mut slope_row = Array2::<f64>::zeros((1, slices.slope.len()));
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let (row_i, motion_i) = expected_information_psi_motion(axis_i, block_states, primary, row)?;
                    let (row_j, motion_j) = expected_information_psi_motion(axis_j, block_states, primary, row)?;
                    let third_i =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &motion_i)?;
                    let third_j =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &motion_j)?;
                    let fourth = self.expected_information_row_fourth(
                        row,
                        block_states,
                        &cache,
                        flexible,
                        &state,
                        &motion_i,
                        &motion_j,
                    )?;
                    let second_design = pair
                        .second_design
                        .as_ref()
                        .map(|map| -> Result<_, String> {
                            let (row_ij, motion_ij) =
                                expected_information_design_motion(map, axis_i, block_states, primary, row)?;
                            let third_ij = self.expected_information_row_third(
                                row,
                                block_states,
                                &cache,
                                flexible,
                                &state,
                                &motion_ij,
                            )?;
                            Ok((row_ij, motion_ij, third_ij))
                        })
                        .transpose()?;
                    self.marginal_design
                        .row_chunk_into(row..row + 1, marginal_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    self.slope_design
                        .row_chunk_into(row..row + 1, slope_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    for primary_axis in 0..primary.total {
                        let mut unit = Array1::<f64>::zeros(primary.total);
                        unit[primary_axis] = 1.0;
                        let first = state.first_directional(&unit);
                        let along_i = state.second_directional(&unit, &motion_i, &third_i);
                        let along_j = state.second_directional(&unit, &motion_j, &third_j);
                        let mut basis = state.third_directional_axis(
                            &motion_i,
                            &motion_j,
                            &third_i,
                            &third_j,
                            &fourth,
                            primary_axis,
                        );
                        let mut pulled = BernoulliBlockHessianAccumulator::new(slices);
                        if let Some((row_ij, motion_ij, third_ij)) = second_design.as_ref() {
                            basis += &state.second_directional(&unit, motion_ij, third_ij);
                            pulled.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                axis_i.block_idx,
                                row_ij,
                                first.row(q_i),
                            )?;
                        }
                        pulled.add_pullback(self, row, slices, primary, &basis);
                        pulled.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            axis_i.block_idx,
                            &row_i,
                            along_j.row(q_i),
                        )?;
                        pulled.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            axis_j.block_idx,
                            &row_j,
                            along_i.row(q_j),
                        )?;
                        pulled.add_psi_psi_outer(
                            axis_i.block_idx,
                            &row_i,
                            axis_j.block_idx,
                            &row_j,
                            first[[q_i, q_j]],
                        );
                        let pulled = pulled.to_dense(slices);
                        for (coefficient_axis, scale) in primary_axis_coefficients(
                            slices,
                            primary,
                            primary_axis,
                            &marginal_row,
                            &slope_row,
                        ) {
                            if scale != 0.0 {
                                axes[coefficient_axis].scaled_add(scale, &pulled);
                            }
                        }
                        if primary_axis == q_i {
                            let mut moved = BernoulliBlockHessianAccumulator::new(slices);
                            moved.add_pullback(self, row, slices, primary, &along_j);
                            moved.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                axis_j.block_idx,
                                &row_j,
                                first.row(q_j),
                            )?;
                            add_design_row_scaled(&mut axes, block_i.start, &row_i, &moved.to_dense(slices));
                            if let Some((row_ij, _, _)) = second_design.as_ref() {
                                let mut moved = BernoulliBlockHessianAccumulator::new(slices);
                                moved.add_pullback(self, row, slices, primary, &first);
                                add_design_row_scaled(&mut axes, block_i.start, row_ij, &moved.to_dense(slices));
                            }
                        }
                        if primary_axis == q_j {
                            let mut moved = BernoulliBlockHessianAccumulator::new(slices);
                            moved.add_pullback(self, row, slices, primary, &along_i);
                            moved.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                axis_i.block_idx,
                                &row_i,
                                first.row(q_i),
                            )?;
                            add_design_row_scaled(&mut axes, block_j.start, &row_j, &moved.to_dense(slices));
                        }
                    }
                }
                Ok(axes)
            },
            |mut left, right| -> Result<_, String> {
                for (left_axis, right_axis) in left.iter_mut().zip(right.iter()) {
                    *left_axis += right_axis;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(empty);
        Ok(accumulated)
    }

    /// `{D²_β I_ψ[direction, e_a]}` for every coefficient axis `a`, the coefficient derivative
    /// along `direction` of [`Self::expected_jeffreys_information_psi_directional_all_axes`].
    /// With `x = J·direction` and the moved design row's motion `ε = e_q·(r_i·direction)`,
    /// primary `k` of a row contributes `D³M[δ_i, x, e_k] + D²M[ε, e_k]` pulled back and
    /// `×(J_i, D²M[x, e_k])`, scattered through `J[k, a]`, and the design row adds
    /// `r_i[a]·D²M[e_q, x]` pulled back on the ψ block. One contracted row fourth per row,
    /// along `(δ_i, x)`.
    pub(super) fn expected_jeffreys_information_psi_directional_second_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        axis: &PsiAxisSpec,
        direction: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        if direction.len() != total {
            return Err(format!(
                "BMS expected information psi second drift expected a direction of length {total}, got {}",
                direction.len()
            ));
        }
        let flexible = self.effective_flex_active(block_states)?;
        self.prewarm_expected_information_fourth(
            block_states,
            &cache,
            flexible,
            "expected Jeffreys information psi second drift",
        )?;
        let psi_block = psi_block_range(slices, axis);
        let block_direction = direction.slice(s![psi_block.clone()]);
        let q = axis.idx_primary;
        let empty = || vec![Array2::<f64>::zeros((total, total)); total];
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut axes = empty();
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                let mut marginal_row = Array2::<f64>::zeros((1, slices.marginal.len()));
                let mut slope_row = Array2::<f64>::zeros((1, slices.slope.len()));
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let (psi_row, motion) = expected_information_psi_motion(axis, block_states, primary, row)?;
                    let along = self.row_primary_direction_from_flat(row, slices, primary, direction)?;
                    let mut design_motion = Array1::<f64>::zeros(primary.total);
                    design_motion[q] = psi_row.dot(&block_direction);
                    let third_motion =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &motion)?;
                    let third_along =
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, &along)?;
                    let third_design = self.expected_information_row_third(
                        row,
                        block_states,
                        &cache,
                        flexible,
                        &state,
                        &design_motion,
                    )?;
                    let fourth = self.expected_information_row_fourth(
                        row,
                        block_states,
                        &cache,
                        flexible,
                        &state,
                        &motion,
                        &along,
                    )?;
                    self.marginal_design
                        .row_chunk_into(row..row + 1, marginal_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    self.slope_design
                        .row_chunk_into(row..row + 1, slope_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    for primary_axis in 0..primary.total {
                        let mut unit = Array1::<f64>::zeros(primary.total);
                        unit[primary_axis] = 1.0;
                        let along_second = state.second_directional(&unit, &along, &third_along);
                        let mut basis = state.third_directional_axis(
                            &motion,
                            &along,
                            &third_motion,
                            &third_along,
                            &fourth,
                            primary_axis,
                        );
                        basis += &state.second_directional(&unit, &design_motion, &third_design);
                        let mut pulled = BernoulliBlockHessianAccumulator::new(slices);
                        pulled.add_pullback(self, row, slices, primary, &basis);
                        pulled.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            axis.block_idx,
                            &psi_row,
                            along_second.row(q),
                        )?;
                        let pulled = pulled.to_dense(slices);
                        for (coefficient_axis, scale) in primary_axis_coefficients(
                            slices,
                            primary,
                            primary_axis,
                            &marginal_row,
                            &slope_row,
                        ) {
                            if scale != 0.0 {
                                axes[coefficient_axis].scaled_add(scale, &pulled);
                            }
                        }
                        if primary_axis == q {
                            let mut moved = BernoulliBlockHessianAccumulator::new(slices);
                            moved.add_pullback(self, row, slices, primary, &along_second);
                            add_design_row_scaled(&mut axes, psi_block.start, &psi_row, &moved.to_dense(slices));
                        }
                    }
                }
                Ok(axes)
            },
            |mut left, right| -> Result<_, String> {
                for (left_axis, right_axis) in left.iter_mut().zip(right.iter()) {
                    *left_axis += right_axis;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(empty);
        Ok(accumulated)
    }

    /// `∂_t` of a rigid row's log-likelihood, primary gradient and Hessian per unit weight at
    /// `t = log σ` of the Gaussian frailty, from the frailty-scale jet the observed route reads;
    /// `second` asks for `∂²_t`.
    fn expected_information_log_frailty_parts(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        state: &ExpectedInformationRow,
        second: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let (objective, gradient, hessian) = self.row_sigma_primary_terms(row, block_states, second)?;
        let weight = state.weight;
        Ok((
            -objective / weight,
            gradient.mapv(|value| -value / weight),
            state.per_unit_weight(hessian),
        ))
    }

    /// `∂_t D_x ∇²ℓ` of a rigid row per unit weight along a primary direction `x`.
    fn expected_information_log_frailty_directional_curvature(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        state: &ExpectedInformationRow,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let (_, hessian) =
            self.row_sigma_primary_directional_terms(row, block_states, &[direction[0], direction[1]])?;
        Ok(state.per_unit_weight(hessian))
    }

    /// A frailty-scale derivative of the expected information on the rigid kernel. `None` without a
    /// Gaussian frailty, or for a flexible fit, whose frailty-scale kernels the observed route does
    /// not provide either.
    ///
    /// With the slot `t = (ℓ_t, ∇ℓ_t, ∇²ℓ_t)` a row contributes `Jᵀ D M[t] J` for the value,
    /// `Jᵀ D²M[t, t] J` for the second derivative, and along a design axis
    /// `Jᵀ D²M[δ, t] J + ×(J_ψ, D M[t])`, where `D_δ D_t ℓ = δ·∇ℓ_t` and `D_δ D_t ∇ℓ = ∇²ℓ_t δ`:
    /// the frailty scale moves no design row.
    fn expected_jeffreys_information_log_frailty(
        &self,
        block_states: &[ParameterBlockState],
        order: LogFrailtyOrder<'_>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.gaussian_frailty_sd.is_none() || self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut accumulator = BernoulliBlockHessianAccumulator::new(slices);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, false, &mut scratch)?
                    else {
                        continue;
                    };
                    let (t_along, t_moved, t_curvature) =
                        self.expected_information_log_frailty_parts(row, block_states, &state, false)?;
                    match order {
                        LogFrailtyOrder::Value => {
                            accumulator.add_pullback(
                                self,
                                row,
                                slices,
                                primary,
                                &state.first_directional_parts(t_along, &t_moved),
                            );
                        }
                        LogFrailtyOrder::Second => {
                            let (tt_along, tt_moved, _) =
                                self.expected_information_log_frailty_parts(row, block_states, &state, true)?;
                            accumulator.add_pullback(
                                self,
                                row,
                                slices,
                                primary,
                                &state.second_directional_parts(
                                    t_along,
                                    state.odds * t_along,
                                    &t_moved,
                                    t_along,
                                    state.odds * t_along,
                                    &t_moved,
                                    tt_along,
                                    &tt_moved,
                                ),
                            );
                        }
                        LogFrailtyOrder::Design(axis) => {
                            let (psi_row, motion) =
                                expected_information_psi_motion(axis, block_states, primary, row)?;
                            accumulator.add_pullback(
                                self,
                                row,
                                slices,
                                primary,
                                &state.second_directional_parts(
                                    state.gradient.dot(&motion),
                                    state.scaled.dot(&motion),
                                    &state.hessian.dot(&motion),
                                    t_along,
                                    state.odds * t_along,
                                    &t_moved,
                                    motion.dot(&t_moved),
                                    &t_curvature.dot(&motion),
                                ),
                            );
                            accumulator.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                axis.block_idx,
                                &psi_row,
                                state.first_directional_parts(t_along, &t_moved).row(axis.idx_primary),
                            )?;
                        }
                    }
                }
                Ok(accumulator)
            },
            |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            },
        )?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(Some(accumulated.to_dense(slices)))
    }

    /// A frailty-scale derivative of the expected information on every coefficient axis, on the
    /// rigid kernel; `None` as for [`Self::expected_jeffreys_information_log_frailty`]. Along
    /// primary `k` a row contributes, scattered through `J[k, a]`:
    ///
    /// ```text
    /// first        D²M[e_k, t]
    /// second       D³M[t, t, e_k]
    /// design       D³M[δ, t, e_k] + ×(J_ψ, D²M[e_k, t]),  plus r[a]·D²M[e_q, t] on the ψ block
    /// directional  D³M[x, t, e_k],  x = J·direction
    /// ```
    fn expected_jeffreys_information_log_frailty_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        order: LogFrailtyAxesOrder<'_>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        if self.gaussian_frailty_sd.is_none() || self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let total = slices.total;
        if let LogFrailtyAxesOrder::Directional(direction) = order
            && direction.len() != total
        {
            return Err(format!(
                "BMS expected information frailty-scale drift expected a direction of length {total}, got {}",
                direction.len()
            ));
        }
        if matches!(
            order,
            LogFrailtyAxesOrder::Design(_) | LogFrailtyAxesOrder::Directional(_)
        ) {
            self.prewarm_expected_information_fourth(
                block_states,
                &cache,
                false,
                "expected Jeffreys information frailty-scale drift",
            )?;
        }
        let empty = || vec![Array2::<f64>::zeros((total, total)); total];
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut axes = empty();
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                let mut marginal_row = Array2::<f64>::zeros((1, slices.marginal.len()));
                let mut slope_row = Array2::<f64>::zeros((1, slices.slope.len()));
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, false, &mut scratch)?
                    else {
                        continue;
                    };
                    let (t_along, t_moved, t_curvature) =
                        self.expected_information_log_frailty_parts(row, block_states, &state, false)?;
                    let t_slot = RowLikelihoodDerivative {
                        along: t_along,
                        scaled: state.odds * t_along,
                        moved: &t_moved,
                        curvature: &t_curvature,
                    };
                    // The other slot and its mixed derivatives with `t`, for the third-order orders.
                    let second_t = match order {
                        LogFrailtyAxesOrder::Second => {
                            Some(self.expected_information_log_frailty_parts(row, block_states, &state, true)?)
                        }
                        _ => None,
                    };
                    let design_motion = match order {
                        LogFrailtyAxesOrder::Design(axis) => {
                            Some(expected_information_psi_motion(axis, block_states, primary, row)?)
                        }
                        _ => None,
                    };
                    let other = match order {
                        LogFrailtyAxesOrder::Design(_) | LogFrailtyAxesOrder::Directional(_) => {
                            let direction = match order {
                                LogFrailtyAxesOrder::Directional(direction) => {
                                    self.row_primary_direction_from_flat(row, slices, primary, direction)?
                                }
                                _ => design_motion
                                    .as_ref()
                                    .map(|(_, motion)| motion.clone())
                                    .unwrap_or_else(|| Array1::zeros(primary.total)),
                            };
                            let moved = state.hessian.dot(&direction);
                            let third =
                                self.expected_information_row_third(row, block_states, &cache, false, &state, &direction)?;
                            let mixed_curvature = self.expected_information_log_frailty_directional_curvature(
                                row,
                                block_states,
                                &state,
                                &direction,
                            )?;
                            let mixed_moved = t_curvature.dot(&direction);
                            Some((direction, moved, third, mixed_moved, mixed_curvature))
                        }
                        _ => None,
                    };
                    self.marginal_design
                        .row_chunk_into(row..row + 1, marginal_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    self.slope_design
                        .row_chunk_into(row..row + 1, slope_row.view_mut())
                        .map_err(|error| error.to_string())?;
                    for primary_axis in 0..primary.total {
                        let unit_moved = state.hessian.column(primary_axis).to_owned();
                        let t_unit_moved = t_curvature.column(primary_axis).to_owned();
                        let unit_with_t = state.second_directional_parts(
                            state.gradient[primary_axis],
                            state.scaled[primary_axis],
                            &unit_moved,
                            t_along,
                            state.odds * t_along,
                            &t_moved,
                            t_moved[primary_axis],
                            &t_unit_moved,
                        );
                        let basis = match (order, second_t.as_ref(), other.as_ref()) {
                            (LogFrailtyAxesOrder::First, _, _) => unit_with_t.clone(),
                            (LogFrailtyAxesOrder::Second, Some((tt_along, tt_moved, tt_curvature)), _) => state
                                .third_directional_axis_parts(
                                    &t_slot,
                                    &t_slot,
                                    &RowLikelihoodDerivative {
                                        along: *tt_along,
                                        scaled: state.odds * *tt_along,
                                        moved: tt_moved,
                                        curvature: tt_curvature,
                                    },
                                    primary_axis,
                                ),
                            (_, _, Some((direction, moved, third, mixed_moved, mixed_curvature))) => state
                                .third_directional_axis_parts(
                                    &RowLikelihoodDerivative {
                                        along: state.gradient.dot(direction),
                                        scaled: state.scaled.dot(direction),
                                        moved,
                                        curvature: third,
                                    },
                                    &t_slot,
                                    &RowLikelihoodDerivative {
                                        along: direction.dot(&t_moved),
                                        scaled: state.odds * direction.dot(&t_moved),
                                        moved: mixed_moved,
                                        curvature: mixed_curvature,
                                    },
                                    primary_axis,
                                ),
                            _ => {
                                return Err(
                                    "BMS expected information frailty-scale drift lost its row slots".to_string(),
                                );
                            }
                        };
                        let mut pulled = BernoulliBlockHessianAccumulator::new(slices);
                        pulled.add_pullback(self, row, slices, primary, &basis);
                        if let (LogFrailtyAxesOrder::Design(axis), Some((psi_row, _))) =
                            (order, design_motion.as_ref())
                        {
                            pulled.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                axis.block_idx,
                                psi_row,
                                unit_with_t.row(axis.idx_primary),
                            )?;
                        }
                        let pulled = pulled.to_dense(slices);
                        for (coefficient_axis, scale) in primary_axis_coefficients(
                            slices,
                            primary,
                            primary_axis,
                            &marginal_row,
                            &slope_row,
                        ) {
                            if scale != 0.0 {
                                axes[coefficient_axis].scaled_add(scale, &pulled);
                            }
                        }
                        // `D_β δ[e_a] = e_q·r[a]` moves primary `q` on the ψ block's axes.
                        if let (LogFrailtyAxesOrder::Design(axis), Some((psi_row, _))) =
                            (order, design_motion.as_ref())
                            && primary_axis == axis.idx_primary
                        {
                            let mut moved = BernoulliBlockHessianAccumulator::new(slices);
                            moved.add_pullback(self, row, slices, primary, &unit_with_t);
                            add_design_row_scaled(
                                &mut axes,
                                psi_block_range(slices, axis).start,
                                psi_row,
                                &moved.to_dense(slices),
                            );
                        }
                    }
                }
                Ok(axes)
            },
            |mut left, right| -> Result<_, String> {
                for (left_axis, right_axis) in left.iter_mut().zip(right.iter()) {
                    *left_axis += right_axis;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(empty);
        Ok(Some(accumulated))
    }

    /// The hyperparameter behind `psi_index` that the expected information moves with: a design
    /// axis, or the log Gaussian frailty scale.
    fn expected_information_axis(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<Option<ExpectedInformationAxis>, String> {
        match hyper_layout.axis(psi_index) {
            Some(crate::custom_family::CustomFamilyHyperAxis::DesignPenalty { .. }) => Ok(self
                .expected_information_psi_axis(hyper_layout, psi_index)?
                .map(ExpectedInformationAxis::Design)),
            Some(crate::custom_family::CustomFamilyHyperAxis::Family { family_axis: 0 })
                if self.gaussian_frailty_sd.is_some() =>
            {
                Ok(Some(ExpectedInformationAxis::LogFrailtyScale))
            }
            _ => Ok(None),
        }
    }

    /// A dispatcher's result as the family's typed answer: the motion, or the refusal naming the
    /// hyperparameter and the cause, which is the log Gaussian frailty scale of a flexible fit or
    /// an axis the family does not move with.
    pub(super) fn expected_information_motion<T>(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_indices: &[usize],
        motion: Option<T>,
    ) -> Result<JeffreysInformationMotion<T>, String> {
        if let Some(motion) = motion {
            return Ok(JeffreysInformationMotion::Published(motion));
        }
        for &psi_index in psi_indices {
            if let Some(ExpectedInformationAxis::LogFrailtyScale) =
                self.expected_information_axis(hyper_layout, psi_index)?
            {
                return Ok(JeffreysInformationMotion::Unpublished {
                    reason: format!(
                        "hyper axis {psi_index} is family axis 0, the log Gaussian frailty scale, and a \
                         flexible score-warp or link-deviation fit supplies no frailty-scale motion of \
                         the expected information, as it supplies no observed frailty-scale derivatives"
                    ),
                });
            }
        }
        Ok(JeffreysInformationMotion::Unpublished {
            reason: format!(
                "hyper axes {psi_indices:?} move neither a marginal or slope design nor the Gaussian \
                 frailty scale of the binary marginal-slope family"
            ),
        })
    }

    /// `∂_ψ I|_β` along hyperparameter `psi_index`; `None` for an axis without a producer.
    pub(super) fn expected_jeffreys_information_hyper_derivative(
        &self,
        block_states: &[ParameterBlockState],
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<Option<Array2<f64>>, String> {
        match self.expected_information_axis(hyper_layout, psi_index)? {
            Some(ExpectedInformationAxis::Design(axis)) => self
                .expected_jeffreys_information_psi_derivative(block_states, &axis)
                .map(Some),
            Some(ExpectedInformationAxis::LogFrailtyScale) => {
                self.expected_jeffreys_information_log_frailty(block_states, LogFrailtyOrder::Value)
            }
            None => Ok(None),
        }
    }

    /// `{∂_ψ D_β I[e_a]|_β}` along hyperparameter `psi_index`.
    pub(super) fn expected_jeffreys_information_hyper_derivative_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        match self.expected_information_axis(hyper_layout, psi_index)? {
            Some(ExpectedInformationAxis::Design(axis)) => self
                .expected_jeffreys_information_psi_directional_all_axes(block_states, &axis)
                .map(Some),
            Some(ExpectedInformationAxis::LogFrailtyScale) => self
                .expected_jeffreys_information_log_frailty_all_axes(block_states, LogFrailtyAxesOrder::First),
            None => Ok(None),
        }
    }

    /// `∂_{ψ_i}∂_{ψ_j} I|_β` along two hyperparameters.
    pub(super) fn expected_jeffreys_information_hyper_second_derivative(
        &self,
        block_states: &[ParameterBlockState],
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<Array2<f64>>, String> {
        match (
            self.expected_information_axis(hyper_layout, psi_i)?,
            self.expected_information_axis(hyper_layout, psi_j)?,
        ) {
            (Some(ExpectedInformationAxis::Design(_)), Some(ExpectedInformationAxis::Design(_))) => {
                match self.expected_information_psi_pair(hyper_layout, psi_i, psi_j)? {
                    Some(pair) => self
                        .expected_jeffreys_information_psi_second_derivative(block_states, &pair)
                        .map(Some),
                    None => Ok(None),
                }
            }
            (Some(ExpectedInformationAxis::Design(axis)), Some(ExpectedInformationAxis::LogFrailtyScale))
            | (Some(ExpectedInformationAxis::LogFrailtyScale), Some(ExpectedInformationAxis::Design(axis))) => {
                self.expected_jeffreys_information_log_frailty(block_states, LogFrailtyOrder::Design(&axis))
            }
            (Some(ExpectedInformationAxis::LogFrailtyScale), Some(ExpectedInformationAxis::LogFrailtyScale)) => {
                self.expected_jeffreys_information_log_frailty(block_states, LogFrailtyOrder::Second)
            }
            _ => Ok(None),
        }
    }

    /// `{∂_{ψ_i}∂_{ψ_j} D_β I[e_a]|_β}` along two hyperparameters.
    pub(super) fn expected_jeffreys_information_hyper_second_derivative_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        match (
            self.expected_information_axis(hyper_layout, psi_i)?,
            self.expected_information_axis(hyper_layout, psi_j)?,
        ) {
            (Some(ExpectedInformationAxis::Design(_)), Some(ExpectedInformationAxis::Design(_))) => {
                match self.expected_information_psi_pair(hyper_layout, psi_i, psi_j)? {
                    Some(pair) => self
                        .expected_jeffreys_information_psi_second_derivative_all_axes(block_states, &pair)
                        .map(Some),
                    None => Ok(None),
                }
            }
            (Some(ExpectedInformationAxis::Design(axis)), Some(ExpectedInformationAxis::LogFrailtyScale))
            | (Some(ExpectedInformationAxis::LogFrailtyScale), Some(ExpectedInformationAxis::Design(axis))) => self
                .expected_jeffreys_information_log_frailty_all_axes(
                    block_states,
                    LogFrailtyAxesOrder::Design(&axis),
                ),
            (Some(ExpectedInformationAxis::LogFrailtyScale), Some(ExpectedInformationAxis::LogFrailtyScale)) => self
                .expected_jeffreys_information_log_frailty_all_axes(block_states, LogFrailtyAxesOrder::Second),
            _ => Ok(None),
        }
    }

    /// `{∂_ψ D²_β I[direction, e_a]|_β}` along hyperparameter `psi_index`.
    pub(super) fn expected_jeffreys_information_hyper_directional_second_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        match self.expected_information_axis(hyper_layout, psi_index)? {
            Some(ExpectedInformationAxis::Design(axis)) => self
                .expected_jeffreys_information_psi_directional_second_all_axes(block_states, &axis, direction)
                .map(Some),
            Some(ExpectedInformationAxis::LogFrailtyScale) => self
                .expected_jeffreys_information_log_frailty_all_axes(
                    block_states,
                    LogFrailtyAxesOrder::Directional(direction),
                ),
            None => Ok(None),
        }
    }

    /// The design axis behind hyperparameter `psi_index`, or `None` for an axis that moves no
    /// design.
    pub(super) fn expected_information_psi_axis(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<Option<PsiAxisSpec>, String> {
        let Some(crate::custom_family::CustomFamilyHyperAxis::DesignPenalty { .. }) =
            hyper_layout.axis(psi_index)
        else {
            return Ok(None);
        };
        let (block_idx, local_idx) = crate::marginal_slope_shared::psi_derivative_location(
            hyper_layout.design_derivative_blocks(),
            psi_index,
        )
        .ok_or_else(|| {
            format!("BMS expected Jeffreys information cannot locate design axis {psi_index}")
        })?;
        self.resolve_psi_axis_spec(hyper_layout.design_derivative_blocks(), block_idx, local_idx)
            .map(Some)
    }

    /// Both design axes of a ψ pair and, when they move one block, its second design map.
    pub(super) fn expected_information_psi_pair(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExpectedInformationPsiPair>, String> {
        let (Some(first), Some(second)) = (
            self.expected_information_psi_axis(hyper_layout, psi_i)?,
            self.expected_information_psi_axis(hyper_layout, psi_j)?,
        ) else {
            return Ok(None);
        };
        let second_design = if first.block_idx == second.block_idx {
            let blocks = hyper_layout.design_derivative_blocks();
            let locate = |psi_index: usize| {
                crate::marginal_slope_shared::psi_derivative_location(blocks, psi_index).ok_or_else(|| {
                    format!("BMS expected Jeffreys information cannot locate design axis {psi_index}")
                })
            };
            let (block, local_i) = locate(psi_i)?;
            let (_, local_j) = locate(psi_j)?;
            let n = self.y.len();
            Some(
                gam_custom_family::resolve_custom_family_x_psi_psi_map(
                    &blocks[block][local_i],
                    &blocks[block][local_j],
                    local_j,
                    n,
                    first.psi_map.ncols(),
                    0..n,
                    "BMS expected information second design",
                    &self.policy,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        Ok(Some(ExpectedInformationPsiPair {
            first,
            second,
            second_design,
        }))
    }

    /// `T[·, ·, direction]` of a row's log-likelihood per unit weight: the flexible row program
    /// or the rigid closed form.
    fn expected_information_row_third(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        flexible: bool,
        state: &ExpectedInformationRow,
        direction: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let third = if flexible {
            self.row_primary_third_contracted(
                row,
                block_states,
                cache,
                Self::row_ctx(cache, row),
                direction,
            )?
        } else {
            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
            let third = self.rigid_row_third_contracted(
                row,
                marginal,
                block_states[1].eta[row],
                direction[0],
                direction[1],
            )?;
            Array2::from_shape_fn((2, 2), |(a, b)| third[a][b])
        };
        Ok(state.per_unit_weight(third))
    }

    /// `Q[·, ·, first, second]` of a row's log-likelihood per unit weight.
    fn expected_information_row_fourth(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        flexible: bool,
        state: &ExpectedInformationRow,
        first: &Array1<f64>,
        second: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let fourth = if flexible {
            self.row_primary_fourth_contracted_ordered(
                row,
                block_states,
                cache,
                Self::row_ctx(cache, row),
                first,
                second,
            )?
        } else {
            let tensor = self.rigid_fourth_full_cached(block_states, cache, row)?;
            let fourth = contract_fourth_full(tensor, first[0], first[1], second[0], second[1]);
            Array2::from_shape_fn((2, 2), |(a, b)| fourth[a][b])
        };
        Ok(state.per_unit_weight(fourth))
    }

    /// Build what a parallel fold of contracted row thirds reads before the fold starts, so no
    /// worker races its lazy construction: the degree-15 cell bundle of a flexible fit.
    fn prewarm_expected_information_third(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        flexible: bool,
        what: &'static str,
    ) -> Result<(), String> {
        if flexible {
            cache.row_primary_hessians.reject_device_cpu_recompute(what)?;
            self.prewarm_flex_cell_bundle(block_states, cache, 15)?;
        }
        Ok(())
    }

    /// [`Self::prewarm_expected_information_third`] for contracted row fourths: the degree-21
    /// cell bundle of a flexible fit, or the rigid fourth table, checked finite once.
    fn prewarm_expected_information_fourth(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        flexible: bool,
        what: &'static str,
    ) -> Result<(), String> {
        if flexible {
            cache.row_primary_hessians.reject_device_cpu_recompute(what)?;
            self.prewarm_flex_cell_bundle(block_states, cache, 21)?;
            return Ok(());
        }
        let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
        if !warmed
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .all(|value| value.is_finite())
        {
            return Err(format!("BMS {what}: the rigid fourth tensor is not finite"));
        }
        Ok(())
    }

    /// The row's expected-information state from the same row kernels the likelihood
    /// evaluates: the rigid closed form, or the flexible row from the cached pin or
    /// streamed again.
    fn expected_information_row_state(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        flexible: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<Option<ExpectedInformationRow>, String> {
        if !flexible {
            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
            let slope = block_states[1].eta[row];
            let (negative_log_likelihood, gradient, hessian) =
                self.rigid_row_kernel_eval(row, marginal, slope)?;
            let gradient = Array1::from_vec(gradient.to_vec());
            let hessian = Array2::from_shape_fn((2, 2), |(a, b)| hessian[a][b]);
            return Ok(ExpectedInformationRow::from_negative_log_likelihood(
                self.weights[row],
                negative_log_likelihood,
                gradient.view(),
                hessian.view(),
            ));
        }
        if let Some(((negative_log_likelihood, gradient), hessian)) =
            Self::cached_row_primary_eval(cache, row).zip(Self::cached_row_primary_hessian(cache, row))
        {
            return Ok(ExpectedInformationRow::from_negative_log_likelihood(
                self.weights[row],
                negative_log_likelihood,
                gradient,
                hessian,
            ));
        }
        let negative_log_likelihood = self.lower_bms_flex_row_order2_with_moments(
            row,
            block_states,
            &cache.primary,
            Self::row_ctx(cache, row),
            cache
                .row_cell_moments
                .as_ref()
                .and_then(|bundle| bundle.row(row, 9)),
            cache.cell_family_forest.as_ref(),
            true,
            scratch,
        )?;
        Ok(ExpectedInformationRow::from_negative_log_likelihood(
            self.weights[row],
            negative_log_likelihood,
            scratch.grad.view(),
            scratch.hess.view(),
        ))
    }

    fn expected_information_pullback(
        &self,
        block_states: &[ParameterBlockState],
        order: ExpectedInformationOrder<'_>,
    ) -> Result<Array2<f64>, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = &cache.slices;
        let primary = &cache.primary;
        let flexible = self.effective_flex_active(block_states)?;
        if flexible {
            cache
                .row_primary_hessians
                .reject_device_cpu_recompute("expected Jeffreys information")?;
            if matches!(
                order,
                ExpectedInformationOrder::Second(..) | ExpectedInformationOrder::ContractedTraceHessian(..)
            ) {
                // The contracted third reads the degree-15 cell bundle; build it before
                // the parallel fold so no worker races its lazy construction.
                self.prewarm_flex_cell_bundle(block_states, &cache, 15)?;
            }
        }
        let accumulated = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.y.len(),
            |rows| -> Result<_, String> {
                let mut accumulator = BernoulliBlockHessianAccumulator::new(slices);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in rows {
                    let Some(state) =
                        self.expected_information_row_state(row, block_states, &cache, flexible, &mut scratch)?
                    else {
                        continue;
                    };
                    let project = |direction: &Array1<f64>| {
                        self.row_primary_direction_from_flat(row, slices, primary, direction)
                    };
                    let third_along = |direction: &Array1<f64>| {
                        self.expected_information_row_third(row, block_states, &cache, flexible, &state, direction)
                    };
                    let matrix = match order {
                        ExpectedInformationOrder::Value => state.information(),
                        ExpectedInformationOrder::First(direction) => {
                            state.first_directional(&project(direction)?)
                        }
                        ExpectedInformationOrder::Second(first, second) => {
                            let first = project(first)?;
                            let second = project(second)?;
                            let third = third_along(&second)?;
                            state.second_directional(&first, &second, &third)
                        }
                        ExpectedInformationOrder::ContractedTraceHessian(weight) => {
                            let dimension = primary.total;
                            let mut weight_in_primary = Array2::<f64>::zeros((dimension, dimension));
                            for column in 0..dimension {
                                let mut unit = Array1::<f64>::zeros(dimension);
                                unit[column] = 1.0;
                                let mut pulled = Array1::<f64>::zeros(slices.total);
                                self.pullback_primary_vector_add_into(
                                    row, slices, primary, &unit, &mut pulled,
                                )?;
                                weight_in_primary
                                    .column_mut(column)
                                    .assign(&project(&weight.dot(&pulled))?);
                            }
                            let scaled_weight = weight_in_primary.dot(&state.scaled);
                            let third = third_along(&scaled_weight)?;
                            state.contracted_trace_hessian(&weight_in_primary, &third)
                        }
                    };
                    accumulator.add_pullback(self, row, slices, primary, &matrix);
                }
                Ok(accumulator)
            },
            |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            },
        )?
        .unwrap_or_else(|| BernoulliBlockHessianAccumulator::new(slices));
        Ok(accumulated.to_dense(slices))
    }
}

#[cfg(test)]
mod expected_information_2922_tests {
    use super::*;
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use std::sync::Arc;

    /// The odds route against the probability route at saturating margins, for both
    /// outcomes: `κ(ℓ)·(dℓ/dη)²` must equal `φ(η)²/(Φ(η)Φ(−η))`, compared in logs
    /// because the weight itself is far below one. `ln Φ` is non-positive on every
    /// branch. Past the underflow of `Φ(−η)` the row is unresolved, and on the
    /// misclassified side past the overflow of `expm1(−ℓ)` the weight is zero; in
    /// both the true weight is below the smallest normal double.
    #[test]
    fn expected_information_odds_route_matches_the_probability_route_2922() {
        let log_pdf = |margin: f64| -0.5 * margin * margin - 0.5 * std::f64::consts::TAU.ln();
        let log_cdf = |margin: f64| gam_math::probability::normal_logcdf_derivatives(margin)[0];
        for margin in [5.0_f64, 10.0, 20.0, 37.0] {
            let log_expected = 2.0 * log_pdf(margin) - log_cdf(margin) - log_cdf(-margin);
            for outcome_sign in [1.0_f64, -1.0] {
                let stack = gam_math::probability::normal_logcdf_derivatives(outcome_sign * margin);
                assert!(
                    stack[0] <= 0.0,
                    "ln Φ({}) = {:.6e} is positive on this branch",
                    outcome_sign * margin,
                    stack[0]
                );
                let slope = outcome_sign * stack[1];
                let odds = expected_information_odds(stack[0])
                    .expect("a margin inside the representable tail is resolved");
                let weight = (odds * slope) * slope;
                assert!(
                    weight > 0.0,
                    "margin {margin} outcome {outcome_sign}: expected weight {weight:.3e} is not positive"
                );
                assert!(
                    (weight.ln() - log_expected).abs() <= 1e-10,
                    "margin {margin} outcome {outcome_sign}: ln weight {:.12e} against ln φ²/(ΦΦ̄) {log_expected:.12e}",
                    weight.ln()
                );
            }
        }
        let misclassified = -40.0_f64;
        let stack = gam_math::probability::normal_logcdf_derivatives(misclassified);
        for outcome_sign in [1.0_f64, -1.0] {
            let odds = expected_information_odds(stack[0])
                .expect("an overflowing complement odds is the zero weight, not unresolved");
            let slope = outcome_sign * stack[1];
            assert!(slope.is_finite(), "the misclassified row gradient {slope:.3e} is not finite");
            let weight = (odds * slope) * slope;
            assert!(
                weight == 0.0,
                "signed margin {misclassified} outcome {outcome_sign}: weight {weight:.3e} is not the zero it rounds to"
            );
        }
        let log_misclassified =
            2.0 * log_pdf(misclassified) - log_cdf(misclassified) - log_cdf(-misclassified);
        assert!(
            log_misclassified < f64::MIN_POSITIVE.ln(),
            "the misclassified row's true weight e^{log_misclassified:.3} is representable"
        );
        let margin = 38.6_f64;
        assert!(
            expected_information_odds(log_cdf(margin)).is_none(),
            "Φ(−{margin}) has underflowed, so the row must be unresolved"
        );
        let log_expected = 2.0 * log_pdf(margin) - log_cdf(margin) - log_cdf(-margin);
        assert!(
            log_expected < f64::MIN_POSITIVE.ln(),
            "the unresolved row's true weight e^{log_expected:.3} is representable"
        );
    }

    /// The six standard-normal rows of `flex_verify_932_tests`' coefficient-surface
    /// gate, with both flexible blocks, or on the rigid kernel.
    fn six_row_fixture(
        flexible: bool,
    ) -> (
        BernoulliMarginalSlopeFamily,
        Array1<f64>,
        Box<dyn Fn(&Array1<f64>) -> Vec<ParameterBlockState>>,
    ) {
        let (mut family, one_row_states) =
            crate::bms::flex_verify_932_tests::standard_normal_flex_fixture();
        let n = 6usize;
        let covariate = Array1::linspace(-0.9, 1.1, n);
        let marginal_x =
            Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { covariate[i] });
        let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { 1.0 } else { 0.5 * covariate[n - 1 - i] }
        });
        family.y = Arc::new(Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]));
        family.weights = Arc::new(Array1::from_vec(vec![0.9, 1.1, 0.8, 1.0, 1.2, 0.7]));
        family.z = Arc::new(Array1::linspace(-1.2, 1.4, n));
        family.marginal_design = DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone()));
        family.slope_design = DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone()));
        let (score_beta, link_beta) = if flexible {
            (one_row_states[2].beta.clone(), one_row_states[3].beta.clone())
        } else {
            family.score_warp = None;
            family.link_dev = None;
            (Array1::zeros(0), Array1::zeros(0))
        };
        let score_dim = score_beta.len();
        let mut beta = Array1::<f64>::zeros(4 + score_dim + link_beta.len());
        beta[0] = 0.18;
        beta[1] = -0.12;
        beta[2] = 0.32;
        beta[3] = 0.08;
        beta.slice_mut(s![4..4 + score_dim]).assign(&score_beta);
        beta.slice_mut(s![4 + score_dim..]).assign(&link_beta);
        let states_at = move |beta: &Array1<f64>| -> Vec<ParameterBlockState> {
            let marginal_beta = beta.slice(s![0..2]).to_owned();
            let slope_beta = beta.slice(s![2..4]).to_owned();
            let mut states = vec![
                ParameterBlockState {
                    eta: marginal_x.dot(&marginal_beta),
                    beta: marginal_beta,
                },
                ParameterBlockState {
                    eta: slope_x.dot(&slope_beta),
                    beta: slope_beta,
                },
            ];
            if flexible {
                states.push(ParameterBlockState {
                    eta: Array1::zeros(n),
                    beta: beta.slice(s![4..4 + score_dim]).to_owned(),
                });
                states.push(ParameterBlockState {
                    eta: Array1::zeros(n),
                    beta: beta.slice(s![4 + score_dim..]).to_owned(),
                });
            }
            states
        };
        (family, beta, Box::new(states_at))
    }

    fn fixture_direction(p: usize, stride: usize, offset: usize, modulus: usize, shift: f64) -> Array1<f64> {
        Array1::from_shape_fn(p, |i| ((i * stride + offset) % modulus) as f64 / modulus as f64 - shift)
    }

    /// A Richardson central difference of `at` compared with `analytic`, on the bar of
    /// `flex_verify_932_tests`' coefficient-surface gate: a resolved difference, agreement
    /// to `1e-5` of the surface scale plus four difference uncertainties, and a surface
    /// with curvature to compare.
    fn difference_misses(
        what: &str,
        analytic: &Array2<f64>,
        at: &dyn Fn(f64) -> Array2<f64>,
    ) -> Vec<String> {
        let step = 1e-3;
        let coarse = (at(step) - at(-step)) / (2.0 * step);
        let fine = (at(0.5 * step) - at(-0.5 * step)) / step;
        let scale = analytic.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let mut misses = Vec::new();
        if !(scale > 1e-6) {
            misses.push(format!("{what} carries no curvature on this fixture ({scale:.3e})"));
        }
        for (((row, column), &want), (&c, &f)) in analytic
            .indexed_iter()
            .zip(coarse.iter().zip(fine.iter()))
        {
            let value = (4.0 * f - c) / 3.0;
            let uncertainty = (f - c).abs() / 3.0;
            let denominator = scale.max(want.abs()).max(value.abs());
            if !(uncertainty <= 0.05 * denominator) {
                misses.push(format!(
                    "{what}[{row}, {column}]: unresolved difference {value:.6e} ± {uncertainty:.3e}"
                ));
            }
            if !((want - value).abs() <= 1e-5 * denominator + 4.0 * uncertainty) {
                misses.push(format!(
                    "{what}[{row}, {column}]: analytic={want:.9e} difference={value:.9e} uncertainty={uncertainty:.3e}"
                ));
            }
        }
        misses
    }

    /// gam#2922 condition (d): the first, second and third coefficient derivatives are
    /// derivatives of the SAME expected information the value path returns, on the
    /// rigid kernel and with both flexible blocks, and that information is positive
    /// semidefinite to its spectrum's rounding band.
    #[test]
    fn expected_information_derivatives_differentiate_the_expected_information_2922() {
        let mut misses = Vec::new();
        for flexible in [true, false] {
            let (family, beta, states_at) = six_row_fixture(flexible);
            let states = states_at(&beta);
            assert_eq!(
                family.effective_flex_active(&states).expect("flex activity"),
                flexible
            );
            let p = beta.len();
            let first_direction = fixture_direction(p, 7, 3, 11, 0.45);
            let second_direction = fixture_direction(p, 5, 1, 13, 0.5);
            let label = if flexible { "flexible" } else { "rigid" };
            let first = family
                .expected_jeffreys_information_directional(&states, &second_direction)
                .expect("D I[v]");
            misses.extend(difference_misses(
                &format!("{label} D I[v]"),
                &first,
                &|t| {
                    family
                        .expected_jeffreys_information(&states_at(&(&beta + &(&second_direction * t))))
                        .expect("displaced expected information")
                },
            ));
            let second = family
                .expected_jeffreys_information_second_directional(
                    &states,
                    &first_direction,
                    &second_direction,
                )
                .expect("D2 I[u, v]");
            misses.extend(difference_misses(
                &format!("{label} D2 I[u, v]"),
                &second,
                &|t| {
                    family
                        .expected_jeffreys_information_directional(
                            &states_at(&(&beta + &(&first_direction * t))),
                            &second_direction,
                        )
                        .expect("displaced D I[v]")
                },
            ));
            let third = family
                .expected_jeffreys_information_third_all_axes(&states, &first_direction, &second_direction)
                .expect("D3 I[u, v, e_a]");
            assert_eq!(third.len(), p);
            for (axis, analytic) in third.iter().enumerate() {
                let mut unit = Array1::<f64>::zeros(p);
                unit[axis] = 1.0;
                misses.extend(difference_misses(
                    &format!("{label} D3 I[u, v, e_{axis}]"),
                    analytic,
                    &|t| {
                        family
                            .expected_jeffreys_information_second_directional(
                                &states_at(&(&beta + &(&unit * t))),
                                &first_direction,
                                &second_direction,
                            )
                            .expect("displaced D2 I[u, v]")
                    },
                ));
            }
            let information = family
                .expected_jeffreys_information(&states)
                .expect("expected information");
            let (eigenvalues, _) =
                gam_linalg::faer_ndarray::FaerEigh::eigh(&information, faer::Side::Lower)
                    .expect("expected information spectrum");
            let values = eigenvalues.to_vec();
            let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&values);
            let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                smallest >= -band,
                "{label} expected information has eigenvalue {smallest:.3e} below its rounding band {band:.3e}"
            );
        }
        assert!(
            misses.is_empty(),
            "{} entries miss their bar:\n{}",
            misses.len(),
            misses.join("\n")
        );
    }

    /// The contracted trace Hessian is `tr(W·D²I[e_a, e_b])` entry by entry: the one-third
    /// form against the pairwise second directional derivative, to rounding of the
    /// entries they accumulate.
    #[test]
    fn expected_information_contracted_trace_hessian_is_the_pairwise_trace_2922() {
        for flexible in [true, false] {
            let (family, beta, states_at) = six_row_fixture(flexible);
            let states = states_at(&beta);
            let p = beta.len();
            let weight = Array2::from_shape_fn((p, p), |(a, b)| {
                ((3 * (a + b) + 5 * a * b) % 17) as f64 / 17.0 - 0.5
            });
            let contracted = family
                .expected_jeffreys_information_contracted_trace_hessian(&states, &weight)
                .expect("contracted trace Hessian");
            let scale = contracted.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
            assert!(
                scale > 1e-6,
                "the contracted trace Hessian carries no curvature on this fixture ({scale:.3e})"
            );
            let label = if flexible { "flexible" } else { "rigid" };
            let mut misses = Vec::new();
            for a in 0..p {
                for b in a..p {
                    let mut unit_a = Array1::<f64>::zeros(p);
                    unit_a[a] = 1.0;
                    let mut unit_b = Array1::<f64>::zeros(p);
                    unit_b[b] = 1.0;
                    let pairwise = family
                        .expected_jeffreys_information_second_directional(&states, &unit_a, &unit_b)
                        .expect("pairwise D2 I");
                    let trace: f64 = weight.iter().zip(pairwise.iter()).map(|(w, d)| w * d).sum();
                    let magnitude: f64 = weight
                        .iter()
                        .zip(pairwise.iter())
                        .map(|(w, d)| (w * d).abs())
                        .sum();
                    // Both routes accumulate the same row terms in different orders, so
                    // they agree to rounding of the summed magnitudes, on the same
                    // relative scale the difference gate reads its agreement.
                    let denominator = scale.max(magnitude).max(contracted[[a, b]].abs());
                    if !((contracted[[a, b]] - trace).abs() <= 1e-10 * denominator) {
                        misses.push(format!(
                            "{label} [{a}, {b}]: contracted={:.12e} pairwise={trace:.12e} magnitude={magnitude:.3e}",
                            contracted[[a, b]]
                        ));
                    }
                }
            }
            assert!(
                misses.is_empty(),
                "{} entries miss their bar:\n{}",
                misses.len(),
                misses.join("\n")
            );
        }
    }

    /// The one-cache all-axes drifts are the per-direction drifts along every coefficient
    /// axis: `D I[e_a]`, and `D²I[v, e_a]` against the per-direction second derivative with
    /// its arguments in the other order, to rounding of the entries they accumulate.
    #[test]
    fn expected_information_all_axes_drifts_are_the_per_axis_drifts_2922() {
        let mut misses = Vec::new();
        for flexible in [true, false] {
            let (family, beta, states_at) = six_row_fixture(flexible);
            let states = states_at(&beta);
            let p = beta.len();
            let direction = fixture_direction(p, 7, 3, 11, 0.45);
            let label = if flexible { "flexible" } else { "rigid" };
            let first = family
                .expected_jeffreys_information_all_axes(&states, None)
                .expect("all-axes D I");
            let second = family
                .expected_jeffreys_information_all_axes(&states, Some(&direction))
                .expect("all-axes D2 I[v, ·]");
            assert_eq!((first.len(), second.len()), (p, p));
            let unit = |axis: usize| Array1::from_shape_fn(p, |i| if i == axis { 1.0 } else { 0.0 });
            let per_first: Vec<Array2<f64>> = (0..p)
                .map(|axis| {
                    family
                        .expected_jeffreys_information_directional(&states, &unit(axis))
                        .expect("per-axis D I")
                })
                .collect();
            let per_second: Vec<Array2<f64>> = (0..p)
                .map(|axis| {
                    family
                        .expected_jeffreys_information_second_directional(&states, &unit(axis), &direction)
                        .expect("per-axis D2 I")
                })
                .collect();
            for (what, batched, per_axis) in [("D I", &first, &per_first), ("D2 I", &second, &per_second)] {
                let scale = per_axis
                    .iter()
                    .flat_map(|matrix| matrix.iter())
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()));
                if !(scale > 1e-6) {
                    misses.push(format!("{label} {what} carries no curvature on this fixture ({scale:.3e})"));
                }
                for (axis, (got_axis, want_axis)) in batched.iter().zip(per_axis.iter()).enumerate() {
                    for ((row, column), &want) in want_axis.indexed_iter() {
                        let got = got_axis[[row, column]];
                        if !((got - want).abs() <= 1e-10 * scale) {
                            misses.push(format!(
                                "{label} {what}[e_{axis}][{row}, {column}]: batched={got:.12e} per-axis={want:.12e}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            misses.is_empty(),
            "{} entries miss their bar:\n{}",
            misses.len(),
            misses.join("\n")
        );
    }

    /// Positive control for the derivative gate: the OBSERVED Hessian drift is not the
    /// derivative of the expected information on the flexible fixture, so a hook that
    /// kept the observed default would fail the pin.
    #[test]
    fn observed_hessian_drift_fails_the_expected_information_gate_2922() {
        let (family, beta, states_at) = six_row_fixture(true);
        let states = states_at(&beta);
        let direction = fixture_direction(beta.len(), 5, 1, 13, 0.5);
        let observed = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("observed D H[v]")
            .expect("the flexible arm publishes D H");
        let misses = difference_misses("observed D H[v] against D I[v]", &observed, &|t| {
            family
                .expected_jeffreys_information(&states_at(&(&beta + &(&direction * t))))
                .expect("displaced expected information")
        });
        assert!(
            !misses.is_empty(),
            "the observed Hessian drift matched the expected information's difference, so the drift pin has no teeth on this fixture"
        );
    }

    /// gam#2922 condition (d) at second order along ψ: `∂_{ψ_i}∂_{ψ_j} I`, its coefficient drift
    /// and the ψ drift's second coefficient derivative are derivatives of the first-order ψ
    /// producers, on the rigid kernel and with both flexible blocks. The pairs are two marginal
    /// axes with a second design map, a marginal and a slope axis, and a slope axis with itself.
    #[test]
    fn expected_information_psi_second_derivatives_differentiate_the_first_2922() {
        let n = 6usize;
        let x_a = Array2::from_shape_fn((n, 2), |(i, j)| 0.3 * ((i + 2 * j) as f64 * 0.41).sin());
        let x_b = Array2::from_shape_fn((n, 2), |(i, j)| 0.25 * ((2 * i + j) as f64 * 0.37).cos());
        let x_ab = Array2::from_shape_fn((n, 2), |(i, j)| 0.2 * ((i + j + 1) as f64 * 0.53).sin());
        let x_c = Array2::from_shape_fn((n, 2), |(i, j)| 0.35 * ((3 * i + j) as f64 * 0.29).cos());
        // Axes 0 and 1 move the marginal design, axis 0 along the second design map `x_ab`
        // as axis 1 moves; axis 2 moves the slope design. `t_ab` displaces axis 0's own map.
        let blocks_at = |t_ab: f64| {
            let mut derivative_a = design_psi_derivative(&x_a + &(&x_ab * t_ab));
            derivative_a.x_psi_psi = Some(vec![Array2::zeros((n, 2)), x_ab.clone()]);
            vec![
                vec![derivative_a, design_psi_derivative(x_b.clone())],
                vec![design_psi_derivative(x_c.clone())],
            ]
        };
        let mut misses = Vec::new();
        for flexible in [true, false] {
            let (family, beta, states_at) = six_row_fixture(flexible);
            let states = states_at(&beta);
            let p = beta.len();
            let label = if flexible { "flexible" } else { "rigid" };
            let axis_at = |blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>], psi: usize| {
                let (block_idx, local_idx) =
                    crate::marginal_slope_shared::psi_derivative_location(blocks, psi)
                        .expect("the fixture axis is a design derivative");
                family
                    .resolve_psi_axis_spec(blocks, block_idx, local_idx)
                    .expect("psi axis spec")
            };
            let base_blocks = blocks_at(0.0);
            for (psi_i, psi_j, block_j, x_j) in [(0usize, 1usize, 0usize, &x_b), (0, 2, 1, &x_c), (2, 2, 1, &x_c)] {
                let (first, second) = (axis_at(&base_blocks, psi_i), axis_at(&base_blocks, psi_j));
                let second_design = (first.block_idx == second.block_idx).then(|| {
                    let (block, local_i) =
                        crate::marginal_slope_shared::psi_derivative_location(&base_blocks, psi_i)
                            .expect("first axis location");
                    let (_, local_j) =
                        crate::marginal_slope_shared::psi_derivative_location(&base_blocks, psi_j)
                            .expect("second axis location");
                    gam_custom_family::resolve_custom_family_x_psi_psi_map(
                        &base_blocks[block][local_i],
                        &base_blocks[block][local_j],
                        local_j,
                        n,
                        first.psi_map.ncols(),
                        0..n,
                        "expected information test second design",
                        &family.policy,
                    )
                    .expect("second design map")
                });
                let pair = ExpectedInformationPsiPair {
                    first,
                    second,
                    second_design,
                };
                let what = format!("{label} d_psi{psi_j} d_psi{psi_i} I");
                let value = family
                    .expected_jeffreys_information_psi_second_derivative(&states, &pair)
                    .expect("d2_psi I");
                misses.extend(difference_misses(&what, &value, &|t| {
                    let (displaced, moved) = displaced_design(&family, &states, block_j, x_j, t);
                    let t_ab = if (psi_i, psi_j) == (0, 1) { t } else { 0.0 };
                    displaced
                        .expected_jeffreys_information_psi_derivative(&moved, &axis_at(&blocks_at(t_ab), psi_i))
                        .expect("displaced d_psi I")
                }));
                let drifts = family
                    .expected_jeffreys_information_psi_second_derivative_all_axes(&states, &pair)
                    .expect("d2_psi D I");
                assert_eq!(drifts.len(), p);
                for (coefficient_axis, analytic) in drifts.iter().enumerate() {
                    let mut unit = Array1::<f64>::zeros(p);
                    unit[coefficient_axis] = 1.0;
                    misses.extend(difference_misses(
                        &format!("{what} D[e_{coefficient_axis}]"),
                        analytic,
                        &|t| {
                            family
                                .expected_jeffreys_information_psi_second_derivative(
                                    &states_at(&(&beta + &(&unit * t))),
                                    &pair,
                                )
                                .expect("shifted d2_psi I")
                        },
                    ));
                }
            }
            let direction = fixture_direction(p, 3, 1, 7, 0.4);
            for psi in [0usize, 2] {
                let axis = axis_at(&base_blocks, psi);
                let analytic = family
                    .expected_jeffreys_information_psi_directional_second_all_axes(&states, &axis, &direction)
                    .expect("d_psi D2 I");
                assert_eq!(analytic.len(), p);
                for (coefficient_axis, analytic) in analytic.iter().enumerate() {
                    misses.extend(difference_misses(
                        &format!("{label} d_psi{psi} D2 I[v, e_{coefficient_axis}]"),
                        analytic,
                        &|t| {
                            family
                                .expected_jeffreys_information_psi_directional_all_axes(
                                    &states_at(&(&beta + &(&direction * t))),
                                    &axis,
                                )
                                .expect("shifted d_psi D I")
                                .swap_remove(coefficient_axis)
                        },
                    ));
                }
            }
        }
        assert!(
            misses.is_empty(),
            "{} entries miss their bar:\n{}",
            misses.len(),
            misses.join("\n")
        );
    }

    /// gam#2922 condition (d) along the log Gaussian frailty scale `t = log σ`: each frailty-scale
    /// derivative of the expected information, alone, twice, mixed with a design axis, and on every
    /// coefficient axis, differentiates the lower-order producer at a displaced σ or β, on the rigid
    /// kernel. A flexible fit publishes none, as the observed route publishes none.
    #[test]
    fn expected_information_log_frailty_derivatives_differentiate_along_the_scale_2922() {
        let mut misses = Vec::new();
        let (family, beta, states_at) = six_row_fixture(false);
        let states = states_at(&beta);
        let p = beta.len();
        let sigma = family
            .gaussian_frailty_sd
            .expect("the fixture carries a Gaussian frailty");
        let displaced = |t: f64| {
            let mut moved = family.clone();
            moved.gaussian_frailty_sd = Some(sigma * t.exp());
            moved
        };
        let x_psi = Array2::from_shape_fn((6, 2), |(i, j)| 0.3 * ((i + 2 * j) as f64 * 0.41).sin());
        let derivative_blocks = vec![vec![design_psi_derivative(x_psi)], Vec::new()];
        let (block_idx, local_idx) =
            crate::marginal_slope_shared::psi_derivative_location(&derivative_blocks, 0)
                .expect("the fixture axis is a design derivative");
        let axis = family
            .resolve_psi_axis_spec(&derivative_blocks, block_idx, local_idx)
            .expect("psi axis spec");
        let published = |order: LogFrailtyOrder<'_>, at: &BernoulliMarginalSlopeFamily, states: &[ParameterBlockState]| {
            at.expected_jeffreys_information_log_frailty(states, order)
                .expect("frailty-scale derivative")
                .expect("a rigid fit with a Gaussian frailty publishes it")
        };
        let published_axes =
            |order: LogFrailtyAxesOrder<'_>, at: &BernoulliMarginalSlopeFamily, states: &[ParameterBlockState]| {
                at.expected_jeffreys_information_log_frailty_all_axes(states, order)
                    .expect("frailty-scale drift")
                    .expect("a rigid fit with a Gaussian frailty publishes it")
            };
        misses.extend(difference_misses(
            "d_t I",
            &published(LogFrailtyOrder::Value, &family, &states),
            &|t| displaced(t).expected_jeffreys_information(&states).expect("displaced I"),
        ));
        // Positive control (i979 pin 1): the observed frailty-scale Hessian motion fails the same gate.
        let cache = family.build_exact_eval_cache_with_order(&states).expect("exact eval cache");
        let mut observed = BernoulliBlockHessianAccumulator::new(&cache.slices);
        for row in 0..family.y.len() {
            let (_, _, hessian) = family
                .row_sigma_primary_terms(row, &states, false)
                .expect("observed frailty-scale terms");
            observed.add_pullback(&family, row, &cache.slices, &cache.primary, &hessian);
        }
        let control = difference_misses(
            "observed d_t H against d_t I",
            &observed.to_dense(&cache.slices),
            &|t| displaced(t).expected_jeffreys_information(&states).expect("displaced I"),
        );
        assert!(
            !control.is_empty(),
            "the observed frailty-scale Hessian motion matched the expected information's difference, so the frailty-scale pin has no teeth on this fixture"
        );
        misses.extend(difference_misses(
            "d2_t I",
            &published(LogFrailtyOrder::Second, &family, &states),
            &|t| published(LogFrailtyOrder::Value, &displaced(t), &states),
        ));
        misses.extend(difference_misses(
            "d_t d_psi I",
            &published(LogFrailtyOrder::Design(&axis), &family, &states),
            &|t| {
                displaced(t)
                    .expected_jeffreys_information_psi_derivative(&states, &axis)
                    .expect("displaced d_psi I")
            },
        ));
        let direction = fixture_direction(p, 3, 1, 7, 0.4);
        let first = published_axes(LogFrailtyAxesOrder::First, &family, &states);
        let second = published_axes(LogFrailtyAxesOrder::Second, &family, &states);
        let mixed = published_axes(LogFrailtyAxesOrder::Design(&axis), &family, &states);
        let along = published_axes(LogFrailtyAxesOrder::Directional(&direction), &family, &states);
        for (coefficient_axis, (((first, second), mixed), along)) in first
            .iter()
            .zip(&second)
            .zip(&mixed)
            .zip(&along)
            .enumerate()
        {
            let mut unit = Array1::<f64>::zeros(p);
            unit[coefficient_axis] = 1.0;
            misses.extend(difference_misses(&format!("d_t D I[e_{coefficient_axis}]"), first, &|t| {
                displaced(t)
                    .expected_jeffreys_information_directional(&states, &unit)
                    .expect("displaced D I")
            }));
            misses.extend(difference_misses(&format!("d2_t D I[e_{coefficient_axis}]"), second, &|t| {
                published_axes(LogFrailtyAxesOrder::First, &displaced(t), &states).swap_remove(coefficient_axis)
            }));
            misses.extend(difference_misses(&format!("d_t d_psi D I[e_{coefficient_axis}]"), mixed, &|t| {
                displaced(t)
                    .expected_jeffreys_information_psi_directional_all_axes(&states, &axis)
                    .expect("displaced d_psi D I")
                    .swap_remove(coefficient_axis)
            }));
            misses.extend(difference_misses(&format!("d_t D2 I[v, e_{coefficient_axis}]"), along, &|t| {
                published_axes(LogFrailtyAxesOrder::First, &family, &states_at(&(&beta + &(&direction * t))))
                    .swap_remove(coefficient_axis)
            }));
        }
        let rigid_layout = crate::custom_family::CustomFamilyHyperLayout::new(
            vec![Vec::new(), Vec::new()],
            vec![0],
            Array1::zeros(1),
        )
        .expect("rigid frailty-scale hyper layout");
        let published = family
            .expected_jeffreys_information_hyper_derivative(&states, &rigid_layout, 0)
            .expect("rigid frailty-scale derivative request");
        assert!(
            matches!(
                family
                    .expected_information_motion(&rigid_layout, &[0], published)
                    .expect("rigid frailty-scale motion"),
                JeffreysInformationMotion::Published(_)
            ),
            "a rigid fit with a Gaussian frailty publishes its frailty-scale derivative"
        );
        let (flexible_family, flexible_beta, flexible_states_at) = six_row_fixture(true);
        let flexible_states = flexible_states_at(&flexible_beta);
        let frailty_layout = crate::custom_family::CustomFamilyHyperLayout::new(
            vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            vec![0],
            Array1::zeros(1),
        )
        .expect("flexible frailty-scale hyper layout");
        assert!(
            matches!(
                frailty_layout.axis(0),
                Some(crate::custom_family::CustomFamilyHyperAxis::Family { family_axis: 0 })
            ),
            "hyper axis 0 of the frailty layout is the family's log frailty scale"
        );
        let derivative = flexible_family
            .expected_jeffreys_information_hyper_derivative(&flexible_states, &frailty_layout, 0)
            .expect("flexible frailty-scale derivative request");
        let second_derivative = flexible_family
            .expected_jeffreys_information_hyper_second_derivative(&flexible_states, &frailty_layout, 0, 0)
            .expect("flexible frailty-scale second derivative request");
        let drifts = flexible_family
            .expected_jeffreys_information_hyper_derivative_all_axes(&flexible_states, &frailty_layout, 0)
            .expect("flexible frailty-scale drift request");
        for (label, refused) in [
            (
                "d_t I",
                matches!(
                    flexible_family.expected_information_motion(&frailty_layout, &[0], derivative),
                    Ok(JeffreysInformationMotion::Unpublished { .. })
                ),
            ),
            (
                "d2_t I",
                matches!(
                    flexible_family.expected_information_motion(&frailty_layout, &[0, 0], second_derivative),
                    Ok(JeffreysInformationMotion::Unpublished { .. })
                ),
            ),
            (
                "d_t D I",
                matches!(
                    flexible_family.expected_information_motion(&frailty_layout, &[0], drifts),
                    Ok(JeffreysInformationMotion::Unpublished { .. })
                ),
            ),
        ] {
            assert!(
                refused,
                "a flexible fit must refuse {label} on the frailty-scale axis with the Unpublished variant"
            );
        }
        assert!(
            misses.is_empty(),
            "{} entries miss their bar:\n{}",
            misses.len(),
            misses.join("\n")
        );
    }

    /// The direction formulas before the derivative-parts refactor, verbatim, for its no-change
    /// control (i979 pin 2).
    fn previous_first_directional(state: &ExpectedInformationRow, direction: &Array1<f64>) -> Array2<f64> {
        let dimension = state.gradient.len();
        let along = state.gradient.dot(direction);
        let moved = state.hessian.dot(direction);
        let mut out = Array2::zeros((dimension, dimension));
        add_symmetric_outer(&mut out, along, &state.scaled, &state.gradient);
        add_symmetric_outer(&mut out, along, &state.scaled, &state.scaled);
        add_symmetric_outer(&mut out, 2.0, &moved, &state.scaled);
        out.mapv_inplace(|value| value * state.weight);
        out
    }

    fn previous_second_directional(
        state: &ExpectedInformationRow,
        first: &Array1<f64>,
        second: &Array1<f64>,
        third_along_second: &Array2<f64>,
    ) -> Array2<f64> {
        let dimension = state.gradient.len();
        let first_along = state.gradient.dot(first);
        let second_along = state.gradient.dot(second);
        let first_scaled = state.scaled.dot(first);
        let second_scaled = state.scaled.dot(second);
        let first_moved = state.hessian.dot(first);
        let second_moved = state.hessian.dot(second);
        let curvature = first.dot(&second_moved);
        let third_moved = third_along_second.dot(first);
        let mut out = Array2::zeros((dimension, dimension));
        add_symmetric_outer(
            &mut out,
            first_along * second_along + curvature,
            &state.scaled,
            &state.gradient,
        );
        add_symmetric_outer(
            &mut out,
            3.0 * first_along * second_along + 2.0 * first_scaled * second_along + curvature,
            &state.scaled,
            &state.scaled,
        );
        add_symmetric_outer(
            &mut out,
            2.0 * (second_along + second_scaled),
            &first_moved,
            &state.scaled,
        );
        add_symmetric_outer(
            &mut out,
            2.0 * (first_along + first_scaled),
            &second_moved,
            &state.scaled,
        );
        add_symmetric_outer(&mut out, 2.0, &third_moved, &state.scaled);
        add_symmetric_outer(&mut out, 2.0 * state.odds, &second_moved, &first_moved);
        out.mapv_inplace(|value| value * state.weight);
        out
    }

    fn previous_third_directional_axis(
        state: &ExpectedInformationRow,
        first: &Array1<f64>,
        second: &Array1<f64>,
        third_along_first: &Array2<f64>,
        third_along_second: &Array2<f64>,
        fourth: &Array2<f64>,
        axis: usize,
    ) -> Array2<f64> {
        let dimension = state.gradient.len();
        let odds = state.odds;
        let (a_x, a_y, a_z) = (
            state.gradient.dot(first),
            state.gradient.dot(second),
            state.gradient[axis],
        );
        let (b_x, b_y, b_z) = (
            state.scaled.dot(first),
            state.scaled.dot(second),
            state.scaled[axis],
        );
        let g_x = state.hessian.dot(first);
        let g_y = state.hessian.dot(second);
        let g_z = state.hessian.column(axis).to_owned();
        let l_xy = first.dot(&g_y);
        let l_xz = g_x[axis];
        let l_yz = g_y[axis];
        let g_xy = third_along_second.dot(first);
        let g_xz = third_along_first.column(axis).to_owned();
        let g_yz = third_along_second.column(axis).to_owned();
        let g_xyz = fourth.column(axis).to_owned();
        let l_xyz = g_xy[axis];
        let mut out = Array2::zeros((dimension, dimension));
        let s = &state.scaled;
        let g = &state.gradient;
        add_symmetric_outer(&mut out, a_x * a_y * a_z, s, g);
        add_symmetric_outer(
            &mut out,
            7.0 * a_x * a_y * a_z + 12.0 * b_x * a_y * a_z + 6.0 * b_x * b_y * a_z,
            s,
            s,
        );
        let mixed = l_xy * a_z + l_xz * a_y + l_yz * a_x;
        let mixed_scaled = l_xy * b_z + l_xz * b_y + l_yz * b_x;
        add_symmetric_outer(&mut out, mixed, s, g);
        add_symmetric_outer(&mut out, 3.0 * mixed + 2.0 * mixed_scaled, s, s);
        for (a_p, b_p, a_q, b_q, g_r) in [
            (a_x, b_x, a_y, b_y, &g_z),
            (a_x, b_x, a_z, b_z, &g_y),
            (a_y, b_y, a_z, b_z, &g_x),
        ] {
            add_symmetric_outer(&mut out, 2.0 * (a_p * a_q + 3.0 * b_p * a_q + 2.0 * b_p * b_q), g_r, s);
        }
        add_symmetric_outer(&mut out, l_xyz, s, g);
        add_symmetric_outer(&mut out, l_xyz, s, s);
        for (l_pq, g_r) in [(l_xy, &g_z), (l_xz, &g_y), (l_yz, &g_x)] {
            add_symmetric_outer(&mut out, 2.0 * (1.0 + odds) * l_pq, g_r, s);
        }
        for (a_p, b_p, g_qr, g_q, g_r) in [
            (a_x, b_x, &g_yz, &g_y, &g_z),
            (a_y, b_y, &g_xz, &g_x, &g_z),
            (a_z, b_z, &g_xy, &g_x, &g_y),
        ] {
            add_symmetric_outer(&mut out, 2.0 * (a_p + b_p), g_qr, s);
            add_symmetric_outer(&mut out, 2.0 * (b_p + odds * b_p), g_q, g_r);
        }
        add_symmetric_outer(&mut out, 2.0, &g_xyz, s);
        add_symmetric_outer(&mut out, 2.0 * odds, &g_xy, &g_z);
        add_symmetric_outer(&mut out, 2.0 * odds, &g_xz, &g_y);
        add_symmetric_outer(&mut out, 2.0 * odds, &g_yz, &g_x);
        out.mapv_inplace(|value| value * state.weight);
        out
    }

    /// gam#2922 i979 pin 2: the direction wrappers over the derivative parts reproduce the previous
    /// formulas bit for bit on every rigid row, for both outcomes and two primary directions in both
    /// orders.
    #[test]
    fn expected_information_direction_wrappers_keep_their_bits_2922() {
        let (family, beta, states_at) = six_row_fixture(false);
        let states = states_at(&beta);
        let cache = family.build_exact_eval_cache_with_order(&states).expect("exact eval cache");
        let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(cache.primary.total);
        let first = Array1::from_vec(vec![0.6, -0.3]);
        let second = Array1::from_vec(vec![-0.2, 0.9]);
        let same_bits = |label: String, new: &Array2<f64>, previous: &Array2<f64>| {
            assert!(
                new.dim() == previous.dim()
                    && new.iter().zip(previous.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{label}: the parts wrapper changed bits\nnew={new:?}\nprevious={previous:?}"
            );
        };
        let mut outcomes = [false; 2];
        for row in 0..family.y.len() {
            let Some(state) = family
                .expected_information_row_state(row, &states, &cache, false, &mut scratch)
                .expect("row state")
            else {
                continue;
            };
            outcomes[usize::from(family.y[row] == 1.0)] = true;
            for (x, y) in [(&first, &second), (&second, &first)] {
                let third_x = family
                    .expected_information_row_third(row, &states, &cache, false, &state, x)
                    .expect("row third along x");
                let third_y = family
                    .expected_information_row_third(row, &states, &cache, false, &state, y)
                    .expect("row third along y");
                let fourth = family
                    .expected_information_row_fourth(row, &states, &cache, false, &state, x, y)
                    .expect("row fourth along x, y");
                same_bits(
                    format!("row {row} first"),
                    &state.first_directional(x),
                    &previous_first_directional(&state, x),
                );
                same_bits(
                    format!("row {row} second"),
                    &state.second_directional(x, y, &third_y),
                    &previous_second_directional(&state, x, y, &third_y),
                );
                for axis in 0..cache.primary.total {
                    same_bits(
                        format!("row {row} third axis {axis}"),
                        &state.third_directional_axis(x, y, &third_x, &third_y, &fourth, axis),
                        &previous_third_directional_axis(&state, x, y, &third_x, &third_y, &fourth, axis),
                    );
                }
            }
        }
        assert!(
            outcomes[0] && outcomes[1],
            "the fixture's resolved rows must cover both outcomes"
        );
    }

    /// The rigid six-row fixture reduced to one row with latent `z`, outcome `y`, unit weight,
    /// marginal predictor `q` and slope `slope`, keeping the fixture's Gaussian frailty.
    fn one_row_rigid_family(
        q: f64,
        slope: f64,
        z: f64,
        y: f64,
    ) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
        let (mut family, _, _) = six_row_fixture(false);
        family.y = Arc::new(Array1::from_vec(vec![y]));
        family.weights = Arc::new(Array1::from_vec(vec![1.0]));
        family.z = Arc::new(Array1::from_vec(vec![z]));
        family.marginal_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((1, 1), 1.0)));
        family.slope_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem((1, 1), 1.0)));
        let states = vec![
            ParameterBlockState {
                beta: Array1::from_vec(vec![q]),
                eta: Array1::from_vec(vec![q]),
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![slope]),
                eta: Array1::from_vec(vec![slope]),
            },
        ];
        (family, states)
    }

    /// Every frailty-scale derivative of the expected information on a rigid family, labelled.
    fn log_frailty_producers(
        family: &BernoulliMarginalSlopeFamily,
        states: &[ParameterBlockState],
    ) -> Vec<(String, Array2<f64>)> {
        let n = family.y.len();
        let derivative_blocks = vec![vec![design_psi_derivative(Array2::from_elem((n, 1), 0.3))], Vec::new()];
        let (block_idx, local_idx) = crate::marginal_slope_shared::psi_derivative_location(&derivative_blocks, 0)
            .expect("the design axis is a design derivative");
        let axis = family
            .resolve_psi_axis_spec(&derivative_blocks, block_idx, local_idx)
            .expect("psi axis spec");
        let direction = Array1::from_vec(vec![0.6, -0.8]);
        let mut out = Vec::new();
        for (label, order) in [
            ("d_t I", LogFrailtyOrder::Value),
            ("d2_t I", LogFrailtyOrder::Second),
            ("d_t d_psi I", LogFrailtyOrder::Design(&axis)),
        ] {
            out.push((
                label.to_string(),
                family
                    .expected_jeffreys_information_log_frailty(states, order)
                    .expect("frailty-scale derivative")
                    .expect("a rigid fit with a Gaussian frailty publishes it"),
            ));
        }
        for (label, order) in [
            ("d_t D I", LogFrailtyAxesOrder::First),
            ("d2_t D I", LogFrailtyAxesOrder::Second),
            ("d_t d_psi D I", LogFrailtyAxesOrder::Design(&axis)),
            ("d_t D2 I", LogFrailtyAxesOrder::Directional(&direction)),
        ] {
            for (coefficient_axis, matrix) in family
                .expected_jeffreys_information_log_frailty_all_axes(states, order)
                .expect("frailty-scale drift")
                .expect("a rigid fit with a Gaussian frailty publishes it")
                .into_iter()
                .enumerate()
            {
                out.push((format!("{label}[e_{coefficient_axis}]"), matrix));
            }
        }
        out
    }

    /// gam#2922 i979 pin 3: the frailty-scale derivatives of the expected information stay finite
    /// in the probit tail for both outcomes, where `κ` grows on the well-classified side and the
    /// frailty-scale slots multiply its powers, and a row whose `expm1(−ℓ)` rounds to zero
    /// contributes exactly nothing. A centred marginal with slopes 5, 10, 20 and 37 on `z = 1` puts
    /// the signed index in the tail; a slope of 40 on the well-classified side leaves the row
    /// unresolved.
    #[test]
    fn expected_information_log_frailty_derivatives_stay_finite_in_the_tail_2922() {
        for slope in [5.0_f64, 10.0, 20.0, 37.0] {
            for y in [1.0_f64, 0.0] {
                let (family, states) = one_row_rigid_family(0.0, slope, 1.0, y);
                for (label, matrix) in log_frailty_producers(&family, &states) {
                    assert!(
                        matrix.iter().all(|value| value.is_finite()),
                        "slope {slope} y {y}: {label} is not finite: {matrix:?}"
                    );
                }
            }
        }
        let (family, states) = one_row_rigid_family(0.0, 40.0, 1.0, 1.0);
        let cache = family.build_exact_eval_cache_with_order(&states).expect("exact eval cache");
        let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(cache.primary.total);
        assert!(
            family
                .expected_information_row_state(0, &states, &cache, false, &mut scratch)
                .expect("row state")
                .is_none(),
            "the well-classified row at slope 40 must be unresolved"
        );
        for (label, matrix) in log_frailty_producers(&family, &states) {
            assert!(
                matrix.iter().all(|value| *value == 0.0),
                "the unresolved row contributes to {label}: {matrix:?}"
            );
        }
    }

    /// A design hyperparameter derivative that moves only the design (no penalty motion).
    fn design_psi_derivative(x_psi: Array2<f64>) -> crate::custom_family::CustomFamilyBlockPsiDerivative {
        let width = x_psi.ncols();
        crate::custom_family::CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi,
            s_psi: Array2::zeros((width, width)),
            s_psi_components: None,
            s_psi_penalty_components: None,
            x_psi_psi: None,
            s_psi_psi: None,
            s_psi_psi_components: None,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }
    }

    /// The six-row fixture with design block `block_idx` (marginal or slope) moved to
    /// `X + t·∂_ψX` at fixed `β`.
    fn displaced_design(
        family: &BernoulliMarginalSlopeFamily,
        states: &[ParameterBlockState],
        block_idx: usize,
        x_psi: &Array2<f64>,
        t: f64,
    ) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
        let mut displaced = family.clone();
        let design = if block_idx == 0 {
            &family.marginal_design
        } else {
            &family.slope_design
        };
        let base = design
            .as_dense_ref()
            .expect("the fixture design is dense")
            .to_owned();
        let moved = &base + &(x_psi * t);
        let mut moved_states = states.to_vec();
        moved_states[block_idx].eta = moved.dot(&moved_states[block_idx].beta);
        let moved = DesignMatrix::Dense(DenseDesignMatrix::from(moved));
        if block_idx == 0 {
            displaced.marginal_design = moved;
        } else {
            displaced.slope_design = moved;
        }
        (displaced, moved_states)
    }

    /// gam#2922 condition (d) along ψ: the explicit ψ-derivative of the expected
    /// information, and of its coefficient drift on every axis, are derivatives of the
    /// SAME expected information along a design motion at fixed `β`, on the rigid kernel
    /// and with both flexible blocks. The observed `∂_ψH` fails the same gate.
    #[test]
    fn expected_information_psi_derivatives_differentiate_along_the_design_motion_2922() {
        let mut misses = Vec::new();
        for flexible in [true, false] {
            let (family, beta, states_at) = six_row_fixture(flexible);
            let states = states_at(&beta);
            let p = beta.len();
            let x_psi = Array2::from_shape_fn((6, 2), |(i, j)| 0.3 * ((i + 2 * j) as f64 * 0.41).sin());
            let derivative_blocks = vec![vec![design_psi_derivative(x_psi.clone())], Vec::new()];
            let (block_idx, local_idx) =
                crate::marginal_slope_shared::psi_derivative_location(&derivative_blocks, 0)
                    .expect("the fixture axis is a design derivative");
            let axis = family
                .resolve_psi_axis_spec(&derivative_blocks, block_idx, local_idx)
                .expect("psi axis spec");
            let label = if flexible { "flexible" } else { "rigid" };
            let value = family
                .expected_jeffreys_information_psi_derivative(&states, &axis)
                .expect("d_psi I");
            misses.extend(difference_misses(&format!("{label} d_psi I"), &value, &|t| {
                let (displaced, moved) = displaced_design(&family, &states, 0, &x_psi, t);
                displaced
                    .expected_jeffreys_information(&moved)
                    .expect("displaced expected information")
            }));
            let drifts = family
                .expected_jeffreys_information_psi_directional_all_axes(&states, &axis)
                .expect("d_psi D I[e_a]");
            assert_eq!(drifts.len(), p);
            for (coefficient_axis, analytic) in drifts.iter().enumerate() {
                let mut unit = Array1::<f64>::zeros(p);
                unit[coefficient_axis] = 1.0;
                misses.extend(difference_misses(
                    &format!("{label} d_psi D I[e_{coefficient_axis}]"),
                    analytic,
                    &|t| {
                        let (displaced, moved) = displaced_design(&family, &states, 0, &x_psi, t);
                        displaced
                            .expected_jeffreys_information_directional(&moved, &unit)
                            .expect("displaced D I[e_a]")
                    },
                ));
            }
            if flexible {
                let cache = family
                    .build_exact_eval_cache(&states)
                    .expect("exact eval cache");
                let observed = family
                    .exact_newton_joint_psi_terms_from_cache_with_options(
                        &states,
                        &derivative_blocks,
                        0,
                        &cache,
                        &crate::custom_family::BlockwiseFitOptions::default(),
                    )
                    .expect("observed psi terms")
                    .expect("the fixture axis publishes psi terms")
                    .hessian_psi;
                let control = difference_misses("observed d_psi H against d_psi I", &observed, &|t| {
                    let (displaced, moved) = displaced_design(&family, &states, 0, &x_psi, t);
                    displaced
                        .expected_jeffreys_information(&moved)
                        .expect("displaced expected information")
                });
                assert!(
                    !control.is_empty(),
                    "the observed d_psi H matched the expected information's difference, so the psi pin has no teeth on this fixture"
                );
            }
        }
        assert!(
            misses.is_empty(),
            "{} entries miss their bar:\n{}",
            misses.len(),
            misses.join("\n")
        );
    }
}

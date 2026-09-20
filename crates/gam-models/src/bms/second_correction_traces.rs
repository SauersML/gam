//! Row-space traces of the outer Hessian's second-order corrections against a
//! logdet factor (gam#2922).
//!
//! The outer Hessian of the Laplace criterion reads the correction
//! `C = D_β H[u] + D²_β H[d_a, d_b]` of each smoothing-parameter pair only
//! through its logdet trace `tr(Fᵀ·C·F)`. The joint Hessian is the pullback
//! `Σ_i J_iᵀ·H_i·J_i` of each row's primary Hessian, so that trace is
//! `Σ_i (T3_i[G_i, u_i] + T4_i[G_i, d_a,i, d_b,i])`, with `T3_i` and `T4_i` the
//! row's third and fourth primary derivatives, `G_i = J_i·F·Fᵀ·J_iᵀ` and
//! `d_i = J_i·d`. Per row, `t_i = T3_i[G_i, ·]` is the third-order trace
//! gradient, and `N_i = T4_i[G_i, ·, ·] = Σ_c T4_i[f_c, f_c, ·, ·]` over the
//! projected factor columns `f_c = J_i·F·e_c`; every pair is then
//! `t_i·u_i + d_a,iᵀ·N_i·d_b,i`. That is one fourth contraction per row and
//! factor column, each at a repeated direction and so needing no symmetrizing
//! second orientation. Forming the drifts instead takes two fourth contractions
//! and one third per row and pair.
//!
//! An empirical flex row with fewer directions than primaries skips `N_i`: its
//! jets take the directions themselves as free axes, so each factor column's
//! contraction is `T4_i[f_c, f_c, d_a, d_b]` directly, carried at `m²` rather
//! than `r²` order-two coefficients.

use super::EmpiricalZGrid;
use super::cell_moment_assembly::empirical_bms_runtime_batch_lanes;
use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::{BernoulliMarginalSlopeRowExactContext, PrimarySlices};
use crate::custom_family::BlockwiseFitOptions;
use gam_math::jet_scalar::DynamicTwoSeedBatch;
use gam_problem::ParameterBlockState;
use ndarray::{Array1, Array2};

impl BernoulliMarginalSlopeFamily {
    /// `tr(Fᵀ·(D_β H[u_i] + D²_β H[d_a, d_b])·F)` for each `i`, with `u_i` the
    /// `i`-th column of `second_modes`, `(a, b) = pairs[i]` and `d_a` the `a`-th
    /// column of `directions`, over the rows (and Horvitz-Thompson weights) the
    /// joint-Hessian derivative operators visit under `options`.
    pub(super) fn batched_second_correction_logdet_traces(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        options: &BlockwiseFitOptions,
        factor: &Array2<f64>,
        second_modes: &Array2<f64>,
        directions: &Array2<f64>,
        pairs: &[(usize, usize)],
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let rank = factor.ncols();
        if factor.nrows() != slices.total
            || second_modes.nrows() != slices.total
            || directions.nrows() != slices.total
        {
            return Err(format!(
                "bernoulli marginal-slope second correction traces: factor rows {}, mode rows {} \
                 and direction rows {} != coefficient dimension {}",
                factor.nrows(),
                second_modes.nrows(),
                directions.nrows(),
                slices.total
            ));
        }
        if second_modes.ncols() != pairs.len() {
            return Err(format!(
                "bernoulli marginal-slope second correction traces: {} second modes for {} pairs",
                second_modes.ncols(),
                pairs.len()
            ));
        }
        if let Some(&(a, b)) = pairs
            .iter()
            .find(|&&(a, b)| a >= directions.ncols() || b >= directions.ncols())
        {
            return Err(format!(
                "bernoulli marginal-slope second correction traces: pair ({a}, {b}) outside {} \
                 directions",
                directions.ncols()
            ));
        }
        if pairs.is_empty() || n == 0 || rank == 0 {
            return Ok(Array1::zeros(pairs.len()));
        }
        // Warm the lazily built caches serially, before the row fan-out, exactly
        // as the per-pair operator builder does (a lazy cache first filled under
        // Rayon can deadlock).
        if self.effective_flex_active(block_states)? {
            self.prewarm_flex_cell_bundle(block_states, cache, 21)?;
        } else {
            let warmed = self.rigid_fourth_full_cached(block_states, cache, 0)?;
            ensure_finite_fourth_full_cache_row(
                warmed,
                "bernoulli marginal-slope second correction traces rigid fourth-cache warm-up",
            )?;
        }
        let mode_columns: Vec<Array1<f64>> = second_modes
            .columns()
            .into_iter()
            .map(|column| column.to_owned())
            .collect();
        let direction_columns: Vec<Array1<f64>> = directions
            .columns()
            .into_iter()
            .map(|column| column.to_owned())
            .collect();
        let weighted_rows = cache.outer_weighted_rows_cached(options, n);
        let m = directions.ncols();
        // An empirical flex row contracts its fourth derivative straight onto the
        // directions whenever they are fewer than the primaries, so each seed's jet
        // carries `m²` instead of `r²` order-two coefficients.
        let project_onto_directions =
            self.effective_flex_active(block_states)? && m < primary.total;
        let started = std::time::Instant::now();
        // A row costs one fourth contraction per factor column, so the fold
        // declares that work and its leaves fan across the pool even at a few
        // hundred rows. The tree is a function of the row count and rank alone.
        let traces = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold_by_work(
            weighted_rows.len(),
            rank,
            |index_range| -> Result<Vec<f64>, String> {
                // Each block runs on a Rayon worker: keep the row kernels' own
                // parallelism sequential so they do not re-fan the pool.
                gam_problem::with_nested_parallel(|| -> Result<Vec<f64>, String> {
                    let r = primary.total;
                    let mut acc = vec![0.0; pairs.len()];
                    let mut projection = vec![0.0; r * rank];
                    let mut seed = Array1::<f64>::zeros(r);
                    for weighted_row in &weighted_rows[index_range] {
                        let row = weighted_row.index;
                        let row_ctx = Self::row_ctx(cache, row);
                        self.row_factor_primary_projection(
                            row,
                            slices,
                            primary,
                            factor,
                            &mut projection,
                        )?;
                        let gram = Self::row_primary_gram_from_projection(r, rank, &projection);
                        let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &gram,
                        )?;
                        let row_directions = direction_columns
                            .iter()
                            .map(|direction| {
                                self.row_primary_direction_from_flat(
                                    row, slices, primary, direction,
                                )
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                        let grid = if project_onto_directions {
                            self.training_row_grid(row)?
                        } else {
                            None
                        };
                        let fourth = match grid {
                            Some(grid) => self.empirical_flex_row_fourth_factor_contraction_along(
                                row,
                                block_states,
                                primary,
                                row_ctx,
                                &grid,
                                &projection,
                                rank,
                                &row_directions,
                            )?,
                            None => {
                                let mut contracted = Array2::<f64>::zeros((r, r));
                                for column in 0..rank {
                                    for (axis, value) in seed.iter_mut().enumerate() {
                                        *value = projection[axis * rank + column];
                                    }
                                    if seed.iter().all(|value| *value == 0.0) {
                                        continue;
                                    }
                                    contracted += &self.row_primary_fourth_contracted_ordered(
                                        row,
                                        block_states,
                                        cache,
                                        row_ctx,
                                        &seed,
                                        &seed,
                                    )?;
                                }
                                let applied: Vec<Array1<f64>> = row_directions
                                    .iter()
                                    .map(|direction| contracted.dot(direction))
                                    .collect();
                                Array2::from_shape_fn((m, m), |(a, b)| {
                                    row_directions[a].dot(&applied[b])
                                })
                            }
                        };
                        for ((value, &(a, b)), mode) in acc.iter_mut().zip(pairs).zip(&mode_columns)
                        {
                            let row_mode =
                                self.row_primary_direction_from_flat(row, slices, primary, mode)?;
                            *value += weighted_row.weight
                                * (trace_gradient.dot(&row_mode) + fourth[[a, b]]);
                        }
                    }
                    Ok(acc)
                })
            },
            |mut left, right| -> Result<Vec<f64>, String> {
                for (left, right) in left.iter_mut().zip(right.iter()) {
                    *left += *right;
                }
                Ok(left)
            },
        )?
        .unwrap_or_else(|| vec![0.0; pairs.len()]);
        log::debug!(
            "[BMS second-correction traces] rows={} p={} rank={} directions={} pairs={} \
             elapsed={:.3}s",
            weighted_rows.len(),
            slices.total,
            rank,
            directions.ncols(),
            pairs.len(),
            started.elapsed().as_secs_f64(),
        );
        Ok(Array1::from_vec(traces))
    }

    /// `Σ_c T4_i[f_c, f_c, d_a, d_b]` over one empirical flex row's projected
    /// factor columns (`projection`, row-major `r × rank`), as the `m × m` matrix
    /// over its projected directions `d_a`.
    ///
    /// The row plan is compiled once, and the jet's free axes are the directions
    /// themselves (`DynamicTwoSeedBatch::seed_direction_pairs_along`): the lanes
    /// are the factor columns, each the repeated seed `(f_c, f_c)`, and each lane's
    /// contracted fourth is already the `m × m` projection.
    fn empirical_flex_row_fourth_factor_contraction_along(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        grid: &EmpiricalZGrid,
        projection: &[f64],
        rank: usize,
        row_directions: &[Array1<f64>],
    ) -> Result<Array2<f64>, String> {
        let r = primary.total;
        let m = row_directions.len();
        if !(row_ctx.intercept.is_finite() && row_ctx.m_a.is_finite() && row_ctx.m_a > 0.0) {
            return Err(
                "non-finite empirical flexible row context in projected fourth contraction".into(),
            );
        }
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h, beta_w) = self.primary_point_components(&point, primary);
        let plan = self.compile_empirical_bms_row_program(
            row,
            primary,
            q,
            b,
            beta_h.as_ref(),
            beta_w.as_ref(),
            row_ctx.intercept,
            grid,
        )?;
        let primary_point = Self::intercept_primary_point(q, b, beta_h.as_ref(), beta_w.as_ref());
        let seeds: Vec<usize> = (0..rank)
            .filter(|&column| (0..r).any(|axis| projection[axis * rank + column] != 0.0))
            .collect();
        let gradients: Vec<Vec<f64>> = (0..r)
            .map(|axis| {
                row_directions
                    .iter()
                    .map(|direction| direction[axis])
                    .collect()
            })
            .collect();
        let lanes = empirical_bms_runtime_batch_lanes(m);
        let mut total = Array2::<f64>::zeros((m, m));
        self.jet_scratch.batch.with(|workspace| -> Result<(), String> {
            for chunk in seeds.chunks(lanes) {
                workspace.reset(chunk.len());
                let vars = workspace.alloc_slice_fill_with(r, |axis| {
                    DynamicTwoSeedBatch::seed_direction_pairs_along(
                        primary_point[axis],
                        &gradients[axis],
                        &workspace,
                        |lane| {
                            let seed = projection[axis * rank + chunk[lane]];
                            (seed, seed)
                        },
                    )
                });
                let jet = plan.evaluate(vars, 4, &workspace)?;
                for lane in 0..chunk.len() {
                    for (value, contribution) in total.iter_mut().zip(jet.contracted_fourth(lane)) {
                        *value += *contribution;
                    }
                }
            }
            Ok(())
        })?;
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::super::flex_verify_932_tests::standard_normal_flex_fixture;
    use super::super::{EmpiricalZGrid, LatentMeasureKind};
    use super::*;
    use crate::custom_family::CustomFamily;
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use std::sync::Arc;

    /// The flex fixture of `flex_verify_932_tests` widened to six rows with
    /// two-column designs, so the marginal and slope pullbacks are not scalars.
    fn six_row_flex_fixture(
        latent_measure: LatentMeasureKind,
        flex: bool,
    ) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
        let (mut family, one_row_states) = standard_normal_flex_fixture();
        let n = 6usize;
        let covariate = Array1::linspace(-0.9, 1.1, n);
        let marginal_x =
            Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { covariate[i] });
        let slope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                0.5 * covariate[n - 1 - i]
            }
        });
        family.y = Arc::new(Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]));
        family.weights = Arc::new(Array1::from_vec(vec![0.9, 1.1, 0.8, 1.0, 1.2, 0.7]));
        family.z = Arc::new(Array1::linspace(-1.2, 1.4, n));
        family.marginal_design = DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone()));
        family.slope_design = DesignMatrix::Dense(DenseDesignMatrix::from(slope_x.clone()));
        family.latent_measure = latent_measure;
        let marginal_beta = Array1::from_vec(vec![0.18, -0.12]);
        let slope_beta = Array1::from_vec(vec![0.32, 0.08]);
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
        if flex {
            states.push(ParameterBlockState {
                eta: Array1::zeros(n),
                beta: one_row_states[2].beta.clone(),
            });
            states.push(ParameterBlockState {
                eta: Array1::zeros(n),
                beta: one_row_states[3].beta.clone(),
            });
        } else {
            family.score_warp = None;
            family.link_dev = None;
        }
        (family, states)
    }

    fn empirical_grid() -> EmpiricalZGrid {
        let nodes = vec![-1.4_f64, -0.6, 0.1, 0.8, 1.5];
        let raw = [0.14_f64, 0.24, 0.28, 0.20, 0.14];
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        EmpiricalZGrid::new(nodes, weights, "second correction traces 2922 grid")
            .expect("valid grid")
    }

    /// Checks the row kernel against the formed drifts: for every pair,
    /// `tr(Fᵀ·(D_β H[u_i] + D²_β H[d_a, d_b])·F)` from the family's dense
    /// directional and mixed derivatives.
    fn assert_traces_match_formed_drifts(
        family: &BernoulliMarginalSlopeFamily,
        states: &[ParameterBlockState],
        label: &str,
    ) {
        assert_eq!(
            family.effective_flex_active(states).expect("flex activity"),
            states.len() == 4,
            "{label}: fixture flex activity"
        );
        let cache = family
            .build_exact_eval_cache(states)
            .expect("exact eval cache");
        let n = family.y.len();
        let p: usize = states.iter().map(|state| state.beta.len()).sum();
        let rank = p - 1;
        let factor = Array2::from_shape_fn((p, rank), |(i, c)| {
            ((i * 5 + c * 3 + 1) % 17) as f64 / 17.0 - 0.45
        });
        let directions = Array2::from_shape_fn((p, 3), |(i, d)| {
            ((i * 7 + d * 11 + 3) % 13) as f64 / 13.0 - 0.5
        });
        let pairs = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (2, 0)];
        let second_modes = Array2::from_shape_fn((p, pairs.len()), |(i, m)| {
            ((i * 3 + m * 5 + 2) % 11) as f64 / 11.0 - 0.4
        });
        let traces = family
            .batched_second_correction_logdet_traces(
                states,
                &cache,
                &BlockwiseFitOptions::default(),
                &factor,
                &second_modes,
                &directions,
                &pairs,
            )
            .expect("row-kernel traces");
        let mut scale = 0.0_f64;
        let mut worst = 0.0_f64;
        let mut report = Vec::new();
        for (idx, &(a, b)) in pairs.iter().enumerate() {
            let first = family
                .exact_newton_joint_hessian_directional_derivative(
                    states,
                    &second_modes.column(idx).to_owned(),
                )
                .expect("D_beta H")
                .expect("the family publishes D_beta H");
            let second = family
                .exact_newton_joint_hessiansecond_directional_derivative(
                    states,
                    &directions.column(a).to_owned(),
                    &directions.column(b).to_owned(),
                )
                .expect("D2_beta H")
                .expect("the family publishes D2_beta H");
            let formed = &first + &second;
            let expected = (factor.t().dot(&formed).dot(&factor)).diag().sum();
            let second_only = (factor.t().dot(&second).dot(&factor)).diag().sum();
            scale = scale.max(expected.abs()).max(second_only.abs());
            let error = (traces[idx] - expected).abs();
            worst = worst.max(error);
            report.push(format!(
                "({a},{b}) kernel={:.15e} formed={expected:.15e} fourth_part={second_only:.3e} \
                 |diff|={error:.3e}",
                traces[idx]
            ));
        }
        eprintln!("#2922 {label}: {}", report.join("; "));
        assert!(
            scale > 1e-6,
            "{label}: the traces carry no curvature ({scale:.3e}), so the comparison would pass \
             on zeros"
        );
        // Both sides accumulate the same n·p² products (each row's p×p pullback
        // against the factor's Gram), in different orders, so they differ by at
        // most γ_(n·p²) of the largest trace.
        let summands = (n * p * p) as f64;
        let bound = summands * f64::EPSILON / (1.0 - summands * f64::EPSILON) * scale;
        assert!(
            worst <= bound,
            "{label}: the row kernel misses the formed drift traces by {worst:.3e} > {bound:.3e}"
        );
    }

    #[test]
    fn flex_empirical_second_correction_traces_match_the_formed_drifts_2922() {
        let (family, states) = six_row_flex_fixture(
            LatentMeasureKind::GlobalEmpirical {
                grid: empirical_grid(),
            },
            true,
        );
        assert_traces_match_formed_drifts(&family, &states, "flex global-empirical");
    }

    #[test]
    fn flex_standard_normal_second_correction_traces_match_the_formed_drifts_2922() {
        let (family, states) = six_row_flex_fixture(LatentMeasureKind::StandardNormal, true);
        assert_traces_match_formed_drifts(&family, &states, "flex standard-normal");
    }

    #[test]
    fn rigid_second_correction_traces_match_the_formed_drifts_2922() {
        let (family, states) = six_row_flex_fixture(LatentMeasureKind::StandardNormal, false);
        assert_traces_match_formed_drifts(&family, &states, "rigid standard-normal");
    }
}

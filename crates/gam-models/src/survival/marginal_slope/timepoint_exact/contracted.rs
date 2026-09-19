//! Flex primary third/fourth contracted exact tensors.
//!
//! Builds the third- and fourth-order directional contractions of the
//! primary-space NLL Hessian on top of the exact timepoint evaluations, and
//! routes the general entry points to the model-selected flex or rigid path.

use super::flex_jet::{
    FlexTimepointBasePack, FlexTimepointDirectionalPack, pack_flex_timepoint_base,
};
use super::*;

/// Direction-independent geometry shared by the single-direction fused Jet3
/// route and the build-once all-axis route.
struct FlexThirdRowGeometry {
    row: usize,
    p: usize,
    qd1: f64,
    q0: f64,
    q1: f64,
    q0_index: usize,
    q1_index: usize,
    a0: f64,
    a1: f64,
    g: f64,
    o_infl: f64,
    beta_h: Option<Array1<f64>>,
    beta_w: Option<Array1<f64>>,
    entry_cached: CachedPartitionCells,
    exit_cached: CachedPartitionCells,
}

impl FlexThirdRowGeometry {
    fn into_base(
        self,
        entry_base: FlexTimepointBasePack,
        exit_base: FlexTimepointBasePack,
    ) -> FlexThirdRowBase {
        FlexThirdRowBase {
            row: self.row,
            p: self.p,
            qd1: self.qd1,
            q0: self.q0,
            q1: self.q1,
            q0_index: self.q0_index,
            q1_index: self.q1_index,
            a0: self.a0,
            a1: self.a1,
            g: self.g,
            o_infl: self.o_infl,
            beta_h: self.beta_h,
            beta_w: self.beta_w,
            entry_cached: self.entry_cached,
            exit_cached: self.exit_cached,
            entry_base,
            exit_base,
        }
    }
}

impl SurvivalMarginalSlopeFamily {
    /// Exact third-order directional contraction for the flexible survival
    /// path.  Returns D_dir H[u,v] where H is the primary-space NLL Hessian.
    pub(crate) fn row_flex_primary_third_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_third_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival third contracted: dir length {} != primary dimension {p}",
                    dir.len()
                ),
            }
            .into());
        }
        if dir.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let geometry = self.prepare_row_flex_third_geometry(
            row,
            block_states,
            &primary,
            super::flex_jet::FLEX_ORDER_FOUR_MOMENT_DEGREE,
        )?;
        let (entry_base, entry_ext, exit_base, exit_ext) =
            super::flex_jet::with_flex_third_jet_arena(|jet_arena| -> Result<_, String> {
                let (entry_base, entry_ext) = self
                    .compute_survival_timepoint_directional_jet_from_cached(
                        geometry.row,
                        &primary,
                        geometry.q0,
                        geometry.q0_index,
                        geometry.a0,
                        geometry.g,
                        geometry.beta_h.as_ref(),
                        geometry.beta_w.as_ref(),
                        geometry.o_infl,
                        &geometry.entry_cached,
                        dir,
                        jet_arena,
                    )?;
                jet_arena.reset();
                let (exit_base, exit_ext) = self
                    .compute_survival_timepoint_directional_jet_from_cached(
                        geometry.row,
                        &primary,
                        geometry.q1,
                        geometry.q1_index,
                        geometry.a1,
                        geometry.g,
                        geometry.beta_h.as_ref(),
                        geometry.beta_w.as_ref(),
                        geometry.o_infl,
                        &geometry.exit_cached,
                        dir,
                        jet_arena,
                    )?;
                Ok((entry_base, entry_ext, exit_base, exit_ext))
            })?;
        if !exit_base.chi.is_finite() || exit_base.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival third contracted row {row}: non-positive chi1={:.3e}",
                    exit_base.chi,
                ),
            }
            .into());
        }
        let base = geometry.into_base(entry_base, exit_base);
        self.row_flex_third_contract_from_packs(&base, dir, &entry_ext, &exit_ext)
    }

    fn prepare_row_flex_third_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
        moment_order: usize,
    ) -> Result<FlexThirdRowGeometry, String> {
        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?.cloned();
        let beta_w = self.flex_link_beta(block_states)?.cloned();
        let o_infl = self.influence_index_offset(row, block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival third contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, _) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h.as_ref(),
            beta_w.as_ref(),
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, _) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h.as_ref(),
            beta_w.as_ref(),
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        let entry_cached = self.build_cached_partition_with_moment_order(
            row,
            primary,
            a0,
            g,
            beta_h.as_ref(),
            beta_w.as_ref(),
            moment_order,
        )?;
        let exit_cached = self.build_cached_partition_with_moment_order(
            row,
            primary,
            a1,
            g,
            beta_h.as_ref(),
            beta_w.as_ref(),
            moment_order,
        )?;

        Ok(FlexThirdRowGeometry {
            row,
            p: primary.total,
            qd1,
            q0,
            q1,
            q0_index: primary.q0,
            q1_index: primary.q1,
            a0,
            a1,
            g,
            o_infl,
            beta_h,
            beta_w,
            entry_cached,
            exit_cached,
        })
    }

    /// Build the direction-independent per-row geometry that
    /// [`Self::row_flex_third_contract_from_base`] reuses across every
    /// coefficient axis of a Jeffreys all-axes sweep.
    ///
    /// The intercept solves, cached partitions, and exact base timepoints
    /// depend only on the row (its `q`-geometry and the current `β`), not on
    /// the contraction direction `dir`. Hoisting them out of the per-axis loop
    /// turns the all-axes flex third contraction from a `p`-fold rebuild into a
    /// build-once + `p` cheap directional contractions — the #979 flex hot path.
    pub(crate) fn build_row_flex_third_base_with_states(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
    ) -> Result<FlexThirdRowBase, String> {
        self.build_row_flex_third_base_with_moment_order(
            row,
            block_states,
            primary,
            super::flex_jet::FLEX_ORDER_FOUR_MOMENT_DEGREE,
        )
    }

    /// [`Self::build_row_flex_third_base_with_states`] with the cell partitions built
    /// through `FLEX_ORDER_FIVE_MOMENT_DEGREE`, the numeric moments an order-five
    /// timepoint jet reads (gam#2893).
    pub(crate) fn build_row_flex_fifth_base_with_states(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
    ) -> Result<FlexThirdRowBase, String> {
        self.build_row_flex_third_base_with_moment_order(
            row,
            block_states,
            primary,
            super::flex_jet::FLEX_ORDER_FIVE_MOMENT_DEGREE,
        )
    }

    fn build_row_flex_third_base_with_moment_order(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
        moment_order: usize,
    ) -> Result<FlexThirdRowBase, String> {
        let geometry =
            self.prepare_row_flex_third_geometry(row, block_states, primary, moment_order)?;

        // The contracted base timepoint is the Jet2 instance of the same
        // single-source expression used by the Jet3/Jet4 contractions.
        let entry = self.compute_survival_timepoint_exact_jet_from_cached(
            geometry.row,
            primary,
            geometry.q0,
            geometry.q0_index,
            geometry.a0,
            geometry.g,
            geometry.beta_h.as_ref(),
            geometry.beta_w.as_ref(),
            geometry.o_infl,
            &geometry.entry_cached,
        )?;
        let exit = self.compute_survival_timepoint_exact_jet_from_cached(
            geometry.row,
            primary,
            geometry.q1,
            geometry.q1_index,
            geometry.a1,
            geometry.g,
            geometry.beta_h.as_ref(),
            geometry.beta_w.as_ref(),
            geometry.o_infl,
            &geometry.exit_cached,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival third contracted row {}: non-positive chi1={:.3e}",
                    geometry.row, exit.chi,
                ),
            }
            .into());
        }

        Ok(geometry.into_base(
            pack_flex_timepoint_base(&entry),
            pack_flex_timepoint_base(&exit),
        ))
    }

    /// Contract the third-order tensor of a row against a single direction,
    /// reusing the direction-independent [`FlexThirdRowBase`]. Only the
    /// directional timepoint extensions and the third-contraction lowering are
    /// recomputed per axis. Bit-identical to the inline path that
    /// [`Self::row_flex_primary_third_contracted_exact`] previously ran.
    pub(crate) fn row_flex_third_contract_from_base(
        &self,
        base: &FlexThirdRowBase,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let p = base.p;
        if dir.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }
        let primary = flex_primary_slices(self);
        let beta_h = base.beta_h.as_ref();
        let beta_w = base.beta_w.as_ref();

        // The directional pack is the Jet3 instance of the canonical timepoint
        // expression.
        let (entry_ext, exit_ext) =
            super::flex_jet::with_flex_third_jet_arena(|jet_arena| -> Result<_, String> {
                let (_, entry_ext) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q0,
                    base.q0_index,
                    base.a0,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.entry_cached,
                    dir,
                    jet_arena,
                )?;
                jet_arena.reset();
                let (_, exit_ext) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q1,
                    base.q1_index,
                    base.a1,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.exit_cached,
                    dir,
                    jet_arena,
                )?;
                Ok((entry_ext, exit_ext))
            })?;

        self.row_flex_third_contract_from_packs(base, dir, &entry_ext, &exit_ext)
    }

    fn row_flex_third_contract_from_packs(
        &self,
        base: &FlexThirdRowBase,
        dir: &Array1<f64>,
        entry_ext: &FlexTimepointDirectionalPack,
        exit_ext: &FlexTimepointDirectionalPack,
    ) -> Result<Array2<f64>, String> {
        let primary = flex_primary_slices(self);
        // #932 single-source: the contracted third `Σ_c ℓ_{abc} dir_c =
        // (D_dir H)[a][b]` is the ε-Hessian channel of the ONE generic flex
        // row-NLL expression (`flex_row_nll`) instantiated at the one-seed jet
        // `Jet3`, seeded from the base + directional timepoint packs.
        self.flex_row_nll_third_contracted(
            base.row,
            &primary,
            base.q1,
            base.qd1,
            dir.as_slice()
                .ok_or_else(|| "third contraction: dir must be contiguous".to_string())?,
            super::flex_jet::FlexThirdPacks {
                entry_base: &base.entry_base,
                exit_base: &base.exit_base,
                entry_ext,
                exit_ext,
            },
        )
    }

    /// Fourth-order directional contraction for the flexible survival path.
    ///
    /// The mixed second-directional timepoint transport is carried exactly
    /// through the implicit intercept solve, the observed-point eta/chi jets,
    /// and the cellwise density-normalization integrand. The direction-independent
    /// row state comes from [`Self::build_row_flex_third_base_with_states`] and the
    /// contraction from [`Self::row_flex_fourth_contract_from_base`].
    pub(crate) fn row_flex_primary_fourth_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_fourth_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir_u.len() != p || dir_v.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival fourth contracted: dir lengths ({},{}) != {p}",
                    dir_u.len(),
                    dir_v.len(),
                ),
            }
            .into());
        }
        if dir_u.iter().all(|v| v.abs() == 0.0) || dir_v.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }
        let base = self.build_row_flex_third_base_with_states(row, block_states, &primary)?;
        self.row_flex_fourth_contract_from_base(&base, dir_u, dir_v)
    }

    /// Contract the fourth-order tensor of a row against two directions, reusing the
    /// direction-independent [`FlexThirdRowBase`]. The intercept solves, cached
    /// partitions and exact base timepoints are the ones the third contraction reads,
    /// so a caller contracting several directions on one row builds them once
    /// (gam#2893). Only the directional and bidirectional timepoint extensions and the
    /// fourth-contraction lowering depend on the directions.
    pub(crate) fn row_flex_fourth_contract_from_base(
        &self,
        base: &FlexThirdRowBase,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let p = base.p;
        if dir_u.iter().all(|v| v.abs() == 0.0) || dir_v.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }
        let primary = flex_primary_slices(self);
        let beta_h = base.beta_h.as_ref();
        let beta_w = base.beta_w.as_ref();

        // Both directional and mixed-directional timepoint extensions instantiate
        // the same expression at Jet3 and Jet4.
        let (entry_ext_u, entry_ext_v, exit_ext_u, exit_ext_v) =
            super::flex_jet::with_flex_third_jet_arena(|jet_arena| -> Result<_, String> {
                let (_, entry_ext_u) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q0,
                    base.q0_index,
                    base.a0,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.entry_cached,
                    dir_u,
                    jet_arena,
                )?;
                jet_arena.reset();
                let (_, entry_ext_v) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q0,
                    base.q0_index,
                    base.a0,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.entry_cached,
                    dir_v,
                    jet_arena,
                )?;
                jet_arena.reset();
                let (_, exit_ext_u) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q1,
                    base.q1_index,
                    base.a1,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.exit_cached,
                    dir_u,
                    jet_arena,
                )?;
                jet_arena.reset();
                let (_, exit_ext_v) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q1,
                    base.q1_index,
                    base.a1,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.exit_cached,
                    dir_v,
                    jet_arena,
                )?;
                Ok((entry_ext_u, entry_ext_v, exit_ext_u, exit_ext_v))
            })?;

        // Bidirectional extensions D_{d1} D_{d2} (η_uv, χ_uv, D_uv).
        let entry_bi = self.compute_survival_timepoint_bidirectional_jet_from_cached(
            base.row,
            &primary,
            base.q0,
            base.q0_index,
            base.a0,
            base.g,
            beta_h,
            beta_w,
            &base.entry_cached,
            dir_u,
            dir_v,
        )?;
        let exit_bi = self.compute_survival_timepoint_bidirectional_jet_from_cached(
            base.row,
            &primary,
            base.q1,
            base.q1_index,
            base.a1,
            base.g,
            beta_h,
            beta_w,
            &base.exit_cached,
            dir_u,
            dir_v,
        )?;

        // #932 single-source: the contracted fourth `Σ_cd ℓ_{abcd} u_c v_d` is
        // the εδ-Hessian channel of the ONE generic flex row-NLL expression
        // (`flex_row_nll`) instantiated at the two-seed jet `Jet4`, seeded from
        // the base + both directional + bidirectional timepoint packs.
        self.flex_row_nll_fourth_contracted(
            base.row,
            &primary,
            base.q1,
            base.qd1,
            dir_u
                .as_slice()
                .ok_or_else(|| "fourth contraction: dir_u must be contiguous".to_string())?,
            dir_v
                .as_slice()
                .ok_or_else(|| "fourth contraction: dir_v must be contiguous".to_string())?,
            super::flex_jet::FlexFourthPacks {
                entry_base: &base.entry_base,
                exit_base: &base.exit_base,
                entry_ext_u: &entry_ext_u,
                exit_ext_u: &exit_ext_u,
                entry_ext_v: &entry_ext_v,
                exit_ext_v: &exit_ext_v,
                entry_bi: &entry_bi,
                exit_bi: &exit_bi,
            },
        )
    }

    /// The contracted fifth `Σ_{cde} ℓ_{abcde} u_c v_d (e_k)_e` of one row for every
    /// primary axis `k`, sharing the direction-independent base and every pack that
    /// reads only `u` and `v`. The base must carry partitions built through
    /// `FLEX_ORDER_FIVE_MOMENT_DEGREE` (gam#2893).
    pub(crate) fn row_flex_fifth_contract_all_primary_axes_from_base(
        &self,
        base: &FlexThirdRowBase,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let p = base.p;
        if dir_u.iter().all(|value| value.abs() == 0.0)
            || dir_v.iter().all(|value| value.abs() == 0.0)
        {
            return Ok(vec![Array2::<f64>::zeros((p, p)); p]);
        }
        let primary = flex_primary_slices(self);
        let beta_h = base.beta_h.as_ref();
        let beta_w = base.beta_w.as_ref();
        let directional = |dir: &Array1<f64>| -> Result<_, String> {
            super::flex_jet::with_flex_third_jet_arena(|jet_arena| -> Result<_, String> {
                let (_, entry) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q0,
                    base.q0_index,
                    base.a0,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.entry_cached,
                    dir,
                    jet_arena,
                )?;
                jet_arena.reset();
                let (_, exit) = self.compute_survival_timepoint_directional_jet_from_cached(
                    base.row,
                    &primary,
                    base.q1,
                    base.q1_index,
                    base.a1,
                    base.g,
                    beta_h,
                    beta_w,
                    base.o_infl,
                    &base.exit_cached,
                    dir,
                    jet_arena,
                )?;
                Ok((entry, exit))
            })
        };
        let bidirectional = |d1: &Array1<f64>, d2: &Array1<f64>| -> Result<_, String> {
            let entry = self.compute_survival_timepoint_bidirectional_jet_from_cached(
                base.row,
                &primary,
                base.q0,
                base.q0_index,
                base.a0,
                base.g,
                beta_h,
                beta_w,
                &base.entry_cached,
                d1,
                d2,
            )?;
            let exit = self.compute_survival_timepoint_bidirectional_jet_from_cached(
                base.row,
                &primary,
                base.q1,
                base.q1_index,
                base.a1,
                base.g,
                beta_h,
                beta_w,
                &base.exit_cached,
                d1,
                d2,
            )?;
            Ok((entry, exit))
        };
        let (entry_ext_u, exit_ext_u) = directional(dir_u)?;
        let (entry_ext_v, exit_ext_v) = directional(dir_v)?;
        let (entry_bi_uv, exit_bi_uv) = bidirectional(dir_u, dir_v)?;
        let u = dir_u
            .as_slice()
            .ok_or_else(|| "fifth contraction: dir_u must be contiguous".to_string())?;
        let v = dir_v
            .as_slice()
            .ok_or_else(|| "fifth contraction: dir_v must be contiguous".to_string())?;
        let mut out = Vec::with_capacity(p);
        for axis in 0..p {
            let mut dir_w = Array1::<f64>::zeros(p);
            dir_w[axis] = 1.0;
            let (entry_ext_w, exit_ext_w) = directional(&dir_w)?;
            let (entry_bi_uw, exit_bi_uw) = bidirectional(dir_u, &dir_w)?;
            let (entry_bi_vw, exit_bi_vw) = bidirectional(dir_v, &dir_w)?;
            let entry_tri = self.compute_survival_timepoint_tridirectional_jet_from_cached(
                base.row,
                &primary,
                base.q0,
                base.q0_index,
                base.a0,
                base.g,
                beta_h,
                beta_w,
                &base.entry_cached,
                [dir_u, dir_v, &dir_w],
            )?;
            let exit_tri = self.compute_survival_timepoint_tridirectional_jet_from_cached(
                base.row,
                &primary,
                base.q1,
                base.q1_index,
                base.a1,
                base.g,
                beta_h,
                beta_w,
                &base.exit_cached,
                [dir_u, dir_v, &dir_w],
            )?;
            let w = dir_w
                .as_slice()
                .ok_or_else(|| "fifth contraction: axis direction must be contiguous".to_string())?;
            out.push(self.flex_row_nll_fifth_contracted(
                base.row,
                &primary,
                base.q1,
                base.qd1,
                [u, v, w],
                super::flex_jet::FlexFifthPacks {
                    entry_base: &base.entry_base,
                    exit_base: &base.exit_base,
                    entry_ext: [&entry_ext_u, &entry_ext_v, &entry_ext_w],
                    exit_ext: [&exit_ext_u, &exit_ext_v, &exit_ext_w],
                    entry_bi_uv: &entry_bi_uv,
                    exit_bi_uv: &exit_bi_uv,
                    entry_bi_uw: &entry_bi_uw,
                    exit_bi_uw: &exit_bi_uw,
                    entry_bi_vw: &entry_bi_vw,
                    exit_bi_vw: &exit_bi_vw,
                    entry_tri: &entry_tri,
                    exit_tri: &exit_tri,
                },
            )?);
        }
        Ok(out)
    }

    pub(crate) fn row_primary_third_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_third_contracted_exact(row, block_states, dir)
        } else {
            self.row_primary_third_contracted(row, block_states, dir.view())
        }
    }

    pub(crate) fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=6 jet helper, in the family's
        // OWN slope frame. Pinning this to `STATIC_SLOPE_PRIMARIES` evaluated
        // the time-constant row program — a different model — whenever the slope
        // varies along follow-up, and truncated the six-primary directions it
        // was handed to four (#2765). Mirrors `row_primary_third_contracted`.
        fn to_array<const P: usize>(r: [[f64; P]; P]) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((P, P));
            for a in 0..P {
                for b in 0..P {
                    out[[a, b]] = r[a][b];
                }
            }
            out
        }
        if self.slope_is_follow_up_varying() {
            Ok(to_array(self.row_primary_fourth_contracted_tower::<
                DYNAMIC_SLOPE_PRIMARIES,
                DynamicSlopeGeometry,
            >(row, block_states, dir_u, dir_v)?))
        } else {
            Ok(to_array(self.row_primary_fourth_contracted_tower::<
                STATIC_SLOPE_PRIMARIES,
                StaticSlopeGeometry,
            >(row, block_states, dir_u, dir_v)?))
        }
    }

    pub(crate) fn row_primary_fourth_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_fourth_contracted_exact(row, block_states, dir_u, dir_v)
        } else {
            self.row_primary_fourth_contracted(row, block_states, dir_u.view(), dir_v.view())
        }
    }

    // ── Pullback through design matrices ──────────────────────────────
}

//! #932-2 cutover: return shapes + support helpers of the test-only hand
//! directional/bidirectional timepoint oracle producers
//! (`timepoint_exact::{directional,bidirectional}_oracle_tests`).
//!
//! The production contracted path (`timepoint_exact::contracted`) reads the Block-10
//! directional/bidirectional packs directly from the single-source `Jet3`/`Jet4`
//! builders, so these structs + the `CoeffSupport` masks and bilinear-jet coefficient
//! helpers are no longer production surface — they live here, in a test-masked
//! module, consumed only by the hand oracle (which pins the jet path via the
//! `flex_jet` `flex_timepoint_inputs_jet{3,4}_*_matches_hand_932` gates) and the
//! `tests.rs` finite-difference witnesses.

use crate::families::cubic_cell_kernel as exact_kernel;
use crate::families::jet_partitions::MultiDirJet;
use crate::families::marginal_slope_shared::{
    eval_coeff4_at, scale_coeff4, CoeffSupport, ObservedDenestedCellPartials,
};
use ndarray::{Array1, Array2};
// `SurvivalMarginalSlopeFamily` + `FlexPrimarySlices` — for the cell-pair helper
// impl below (moved from `contraction.rs`). Imported explicitly (not `super::*`,
// which would shadow the `CoeffSupport`/`MultiDirJet` imports above and trip the
// unused-import lint).
use super::{CachedCellEntry, FlexPrimarySlices, SurvivalMarginalSlopeFamily};

/// Reconstruct the sign-flipped cubic cell the hand oracle integrates against,
/// from the cached partition cell. Production no longer stores it on
/// `CachedCellEntry` (the jet timepoint path never reads it), so the oracle
/// rebuilds it here by negating the cubic coefficients — byte-identical to the
/// `neg_cell` the cache builder formerly carried.
pub(crate) fn neg_cell_of(entry: &CachedCellEntry) -> exact_kernel::DenestedCubicCell {
    let cell = entry.partition_cell.cell;
    exact_kernel::DenestedCubicCell {
        left: cell.left,
        right: cell.right,
        c0: -cell.c0,
        c1: -cell.c1,
        c2: -cell.c2,
        c3: -cell.c3,
    }
}

// #932-2 cutover: the `MultiDirJet`-coefficient polynomial ops for the hand oracle
// (moved from `poly_arith.rs`, where they were dead in the non-test lib build after
// the production flex path stopped using `MultiDirJet`). Bodies byte-identical.
pub(crate) fn poly_add_jets(lhs: &[MultiDirJet], rhs: &[MultiDirJet]) -> Vec<MultiDirJet> {
    let count = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let left = lhs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| MultiDirJet::zero(2));
        let right = rhs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| MultiDirJet::zero(2));
        out.push(left.add(&right));
    }
    out
}

pub(crate) fn poly_scale_jets(poly: &[MultiDirJet], scale: &MultiDirJet) -> Vec<MultiDirJet> {
    poly.iter().map(|coeff| coeff.mul(scale)).collect()
}

pub(crate) fn poly_mul_jets(lhs: &[MultiDirJet], rhs: &[MultiDirJet]) -> Vec<MultiDirJet> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let mut out = vec![MultiDirJet::zero(2); lhs.len() + rhs.len() - 1];
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            let prod = left.mul(right);
            out[i + j] = out[i + j].add(&prod);
        }
    }
    out
}

pub(crate) fn poly_coeff_mask(poly: &[MultiDirJet], mask: usize) -> Vec<f64> {
    poly.iter().map(|coeff| coeff.coeff(mask)).collect()
}

/// `g+h+w` coefficient-support mask for the hand oracle (slope + score-warp +
/// link-dev primaries active).
pub(crate) const COEFF_SUPPORT_GHW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: true,
    include_w: true,
};

/// `g+w` coefficient-support mask for the hand oracle (slope + link-dev primaries
/// active, score-warp inactive).
pub(crate) const COEFF_SUPPORT_GW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: false,
    include_w: true,
};

/// Scalar composite bilinear jet `base + (da·ad1+fixed_d1)ε + (da·ad2+fixed_d2)δ +
/// (…)εδ` — the per-coefficient second-directional Taylor the hand bidirectional
/// oracle composes (intercept-chain `da/daa` × edge-motion `ad*` plus the fixed
/// channel partials `fixed_d*`/`da_d*`).
pub(crate) fn scalar_composite_bilinear(
    base: f64,
    da: f64,
    daa: f64,
    fixed_d1: f64,
    fixed_d2: f64,
    fixed_d12: f64,
    da_d1: f64,
    da_d2: f64,
    ad1: f64,
    ad2: f64,
    ad12: f64,
) -> MultiDirJet {
    MultiDirJet::bilinear(
        base,
        da * ad1 + fixed_d1,
        da * ad2 + fixed_d2,
        da * ad12 + daa * ad1 * ad2 + da_d1 * ad2 + da_d2 * ad1 + fixed_d12,
    )
}

/// A 4-coefficient cell polynomial as bilinear jets from explicit base / d1 / d2 /
/// d12 channels.
pub(crate) fn coeff4_fixed_bilinear(
    base: &[f64; 4],
    d1: &[f64; 4],
    d2: &[f64; 4],
    d12: &[f64; 4],
) -> Vec<MultiDirJet> {
    (0..4)
        .map(|k| MultiDirJet::bilinear(base[k], d1[k], d2[k], d12[k]))
        .collect()
}

/// A 4-coefficient cell polynomial as composite bilinear jets (per-`k`
/// [`scalar_composite_bilinear`]).
pub(crate) fn coeff4_composite_bilinear(
    base: &[f64; 4],
    da: &[f64; 4],
    daa: &[f64; 4],
    fixed_d1: &[f64; 4],
    fixed_d2: &[f64; 4],
    fixed_d12: &[f64; 4],
    da_d1: &[f64; 4],
    da_d2: &[f64; 4],
    ad1: f64,
    ad2: f64,
    ad12: f64,
) -> Vec<MultiDirJet> {
    (0..4)
        .map(|k| {
            scalar_composite_bilinear(
                base[k],
                da[k],
                daa[k],
                fixed_d1[k],
                fixed_d2[k],
                fixed_d12[k],
                da_d1[k],
                da_d2[k],
                ad1,
                ad2,
                ad12,
            )
        })
        .collect()
}

/// Directional extensions of a timepoint's exact quantities, contracted with a
/// single direction — the pieces the hand third-order NLL contraction oracle composes.
pub(crate) struct SurvivalFlexTimepointDirectionalExact {
    pub(crate) eta_uv_dir: Array2<f64>,
    pub(crate) eta_u_dir: Array1<f64>,
    pub(crate) chi_u_dir: Array1<f64>,
    pub(crate) chi_uv_dir: Array2<f64>,
    pub(crate) d_u_dir: Array1<f64>,
    pub(crate) d_uv_dir: Array2<f64>,
    pub(crate) a_uv_dir: Array2<f64>,
}

/// Mixed second-directional extensions `D_{d1} D_{d2}` of a timepoint's exact
/// quantities — the hand fourth-order NLL contraction oracle's bidirectional pack.
pub(crate) struct SurvivalFlexTimepointBiDirectionalExact {
    pub(crate) eta_uv_uv: Array2<f64>,
    pub(crate) chi_uv_uv: Array2<f64>,
    pub(crate) d_uv_uv: Array2<f64>,
}

// #932-2 cutover: the slope-coupling cell-pair coefficient helpers used ONLY by
// the test-only hand directional/bidirectional oracle. Moved here from
// `contraction.rs`, where they were orphaned (zero production consumer) once the
// production contracted path cut over to the `Jet3`/`Jet4` builders. In survival
// the only b-coupling is through the slope `g`, so each is nonzero only when
// `u==g` or `v==g`.
impl SurvivalMarginalSlopeFamily {
    /// Second-order cross-coefficient for parameter pair (u, v) from the
    /// b-family.  Nonzero only when u==g or v==g.
    pub(crate) fn cell_pair_second_coeff(
        &self,
        primary: &FlexPrimarySlices,
        coeff_bu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_bu[v]
        } else if v == primary.g {
            coeff_bu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Third-order a-cross-coefficient for parameter pair (u, v) from the
    /// ab-family. Nonzero only when u==g or v==g.
    pub(crate) fn cell_pair_third_coeff_a(
        &self,
        primary: &FlexPrimarySlices,
        coeff_abu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_abu[v]
        } else if v == primary.g {
            coeff_abu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Third cell-coefficient cross `∂³c/∂u∂v∂(dir)` contracted with a direction
    /// `dir`, accumulated into `out` (scaled by `sign`). Carried by the `bbu`
    /// family (`∂²/∂b² ∂param`): `u==g,v==g` → `Σ_{c'} coeff_bbu[c']·dir[c']`;
    /// `u==g,v≠g` → `coeff_bbu[v]·dir[g]`; `v==g,u≠g` → `coeff_bbu[u]·dir[g]`;
    /// else `0` (gam#1195).
    pub(crate) fn add_cell_pair_third_coeff_dir(
        &self,
        primary: &FlexPrimarySlices,
        coeff_bbu: &[[f64; 4]],
        u: usize,
        v: usize,
        dir: &Array1<f64>,
        sign: f64,
        out: &mut [f64; 4],
    ) {
        let g = primary.g;
        if u == g && v == g {
            for (c, &dir_c) in dir.iter().enumerate() {
                if dir_c == 0.0 {
                    continue;
                }
                for k in 0..4 {
                    out[k] += sign * coeff_bbu[c][k] * dir_c;
                }
            }
        } else if u == g {
            let dir_g = dir[g];
            if dir_g != 0.0 {
                for k in 0..4 {
                    out[k] += sign * coeff_bbu[v][k] * dir_g;
                }
            }
        } else if v == g {
            let dir_g = dir[g];
            if dir_g != 0.0 {
                for k in 0..4 {
                    out[k] += sign * coeff_bbu[u][k] * dir_g;
                }
            }
        }
    }

    /// The FIXED g×h / g×w second-partial of the observed `eta` (the hand timepoint
    /// Hessian's fixed cross channel) — test-only hand oracle; the production jet path
    /// single-sources this through `flex_jet::cell_coeff_jets`.
    pub(crate) fn observed_fixed_eta_second_partial(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        row: usize,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(eval_coeff4_at(&obs.dc_dbb, z_obs));
        }
        if u == primary.g {
            if let Some(h_range) = primary.h.as_ref()
                && v >= h_range.start
                && v < h_range.end
            {
                let local_idx = v - h_range.start;
                return Ok(eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                        scale,
                    ),
                    z_obs,
                ));
            }
            if let Some(w_range) = primary.w.as_ref()
                && v >= w_range.start
                && v < w_range.end
            {
                let local_idx = v - w_range.start;
                let runtime = self
                    .link_dev
                    .as_ref()
                    .ok_or_else(|| "missing survival link runtime".to_string())?;
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let (_, dc_bw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                return Ok(eval_coeff4_at(&scale_coeff4(dc_bw, scale), z_obs));
            }
        }
        if v == primary.g {
            return self
                .observed_fixed_eta_second_partial(primary, obs, row, v, u, z_obs, u_obs, a, b);
        }
        Ok(0.0)
    }

    /// The FIXED g×w second-partial of the observed `chi` (`∂eta/∂a`) — test-only hand
    /// oracle; the production jet path single-sources this through
    /// `flex_jet::cell_chi_poly_jets`.
    pub(crate) fn observed_fixed_chi_second_partial(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(eval_coeff4_at(&obs.dc_dabb, z_obs));
        }
        if u == primary.g
            && let Some(w_range) = primary.w.as_ref()
            && v >= w_range.start
            && v < w_range.end
        {
            let local_idx = v - w_range.start;
            let runtime = self
                .link_dev
                .as_ref()
                .ok_or_else(|| "missing survival link runtime".to_string())?;
            let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
            let (_, dc_abw, _) = exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
            return Ok(eval_coeff4_at(&scale_coeff4(dc_abw, scale), z_obs));
        }
        if v == primary.g {
            return self.observed_fixed_chi_second_partial(primary, obs, v, u, z_obs, u_obs, a, b);
        }
        Ok(0.0)
    }
}

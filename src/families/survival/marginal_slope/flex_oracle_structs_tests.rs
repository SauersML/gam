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

use crate::families::jet_partitions::MultiDirJet;
use crate::families::marginal_slope_shared::CoeffSupport;
use ndarray::{Array1, Array2};
// `SurvivalMarginalSlopeFamily` + `FlexPrimarySlices` — same resolution as the
// `contraction.rs` impl the cell-pair helpers below were moved from.
use super::*;

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
}

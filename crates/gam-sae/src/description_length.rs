//! Description-length (REML-as-MDL) reporting surface (#2085).
//!
//! The REML negative log-evidence a manifold-SAE / dictionary fit already computes
//! IS a description length: with `v` in nats, `v / ln 2` is bits, and its terms
//! decompose as
//!
//! * **code** bits — the data-fit / distortion term (`loss.total`), the rate to
//!   transmit each firing's coefficients at the achieved distortion;
//! * **selection** bits — the assignment-sparsity term, `log₂ C(G, k)`, naming
//!   which `k` of `G` atoms fired;
//! * **dictionary** bits — the effective-parameter term `½log|XᵀX + S| − occam`,
//!   the amortised cost of storing the decoder.
//!
//! (`SaeManifoldTerm::reml_criterion` forms `v = loss.total + extra_penalty +
//! ½log_det − occam`; construction.rs owns those internals — this module CONSUMES
//! their outputs and never recomputes the fit.)
//!
//! This module is the permanent gam surface for that accounting, ported from the
//! hand-verified `Manifold-SAE experiments/mdl_ladder/mdl.py` reference (incl. its
//! selection-asymmetry fix): a rate-distortion [`score`] of a [`Featurizer`] at a
//! stated distortion floor, and a [`crossover_firings`] comparison of two
//! featurizers → Δbits/token and the crossover firing count `f*` (with the
//! selection-bits delta charged whenever the two configs differ in `(G, k)`, and
//! `f* = ∞` flagged when the richer model never pays — the line/control case). The
//! [`DescriptionLength::from_criterion_nats`] side converts a fit's OWN criterion
//! terms to bits so a test can assert the surface reconciles with the criterion
//! exactly (no parallel accounting drift).

use std::f64::consts::LN_2;

/// Bits to code one Gaussian scalar of variance `signal_var` to per-sample MSE
/// `delta2`: the numerically-kind rate `½log₂(1 + σ²/δ²)` (≥ 0, finite at low SNR;
/// agrees with the high-rate `½log₂(σ²/δ²)` to O(1) bit once `σ² ≫ δ²`).
pub fn scalar_rate_bits(signal_var: f64, delta2: f64) -> f64 {
    if delta2 <= 0.0 {
        return f64::INFINITY;
    }
    0.5 * (1.0 + signal_var.max(0.0) / delta2).log2()
}

/// `log₂ C(G, k)`: bits to name which `k` of `G` dictionary atoms fired. Computed
/// as `Σ_{i=1..k} log₂((G−k+i)/i)` so it never overflows a binomial (exact, and
/// `k` is small in practice). Zero when `G ≤ 0` or `k ≤ 0`; `k` is capped at `G`.
pub fn selection_bits(g_dict: i64, k_active: i64) -> f64 {
    if g_dict <= 0 || k_active <= 0 {
        return 0.0;
    }
    let k = k_active.min(g_dict);
    let mut bits = 0.0;
    for i in 1..=k {
        bits += ((g_dict - k + i) as f64 / i as f64).log2();
    }
    bits
}

/// Rate (bits/sample) of the optimal linear (reverse-water-filling) code of a
/// Gaussian source with covariance eigenvalues `eigs`, coded to total MSE
/// `delta2`. Returns `(total_rate_bits, per_coordinate_bits)`. This is the
/// best a LINEAR featurizer can do at that distortion — the block/direction lower
/// bound a chart must beat.
pub fn reverse_water_filling(eigs: &[f64], delta2: f64) -> (f64, Vec<f64>) {
    if eigs.is_empty() {
        return (0.0, Vec::new());
    }
    let max_e = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (mut lo, mut hi) = (0.0f64, max_e.max(delta2));
    for _ in 0..200 {
        let theta = 0.5 * (lo + hi);
        let dist: f64 = eigs.iter().map(|&e| e.min(theta)).sum();
        if dist > delta2 {
            hi = theta;
        } else {
            lo = theta;
        }
    }
    let theta = 0.5 * (lo + hi);
    let per: Vec<f64> = eigs
        .iter()
        .map(|&e| (0.5 * (e / theta).log2()).max(0.0))
        .collect();
    (per.iter().sum(), per)
}

/// One rung of the description-length ladder — a featurizer's reporting inputs.
///
/// `coded_var` are the per-coordinate signal variances of the `m` coefficients
/// emitted per firing (`m = coded_var.len()`): a direction has `m = 1`, a `b`-block
/// `m = b`, a `d`-chart `m = d`. `n_params` is the DICTIONARY (decoder) scalar
/// count. `total_var` and `ev` fix the achieved residual `(1−ev)·total_var` for
/// the feasibility check.
#[derive(Clone, Debug)]
pub struct Featurizer {
    pub name: String,
    pub kind: String,
    pub coded_var: Vec<f64>,
    pub n_params: i64,
    pub ev: f64,
    pub total_var: f64,
    pub n_tokens: i64,
    pub n_firings: i64,
    pub g_dict: i64,
    pub k_active: i64,
}

impl Featurizer {
    pub fn m(&self) -> usize {
        self.coded_var.len()
    }
    pub fn residual(&self) -> f64 {
        (1.0 - self.ev) * self.total_var
    }
}

/// The scored description length of one featurizer at a stated distortion floor.
#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub name: String,
    pub kind: String,
    pub coded_dim_m: usize,
    pub code_bits_per_firing: f64,
    pub code_coeff_bits_per_firing: f64,
    pub selection_bits_per_firing: f64,
    pub n_params: i64,
    pub l_param_bits: f64,
    pub dict_bits: f64,
    pub code_bits_total: f64,
    pub total_bits: f64,
    pub bits_per_token: f64,
    pub residual_achieved: f64,
    pub distortion_floor: f64,
    pub distortion_infeasible: bool,
}

/// Bits/token description length for `feat` at per-token distortion floor `delta2`.
///
/// `l_param_bits` is the cost to store one dictionary scalar. `None` selects the
/// distortion-matched precision (a decoder weight quantised to the same per-scalar
/// rate as a code coefficient) = the mean per-coefficient code rate; pass a value
/// (e.g. 16 for fp16) to override.
pub fn score(feat: &Featurizer, delta2: f64, l_param_bits: Option<f64>) -> ScoreRow {
    let code_coeff: f64 = feat
        .coded_var
        .iter()
        .map(|&v| scalar_rate_bits(v, delta2))
        .sum();
    let sel = selection_bits(feat.g_dict, feat.k_active);
    let code_per_firing = code_coeff + sel;
    let m = feat.m();
    let l_param = l_param_bits.unwrap_or_else(|| {
        if m > 0 {
            code_coeff / m as f64
        } else {
            scalar_rate_bits(feat.total_var, delta2)
        }
    });
    let dict_bits = feat.n_params as f64 * l_param;
    let code_total = code_per_firing * feat.n_firings as f64;
    let total = code_total + dict_bits;
    let residual = feat.residual();
    ScoreRow {
        name: feat.name.clone(),
        kind: feat.kind.clone(),
        coded_dim_m: m,
        code_bits_per_firing: code_per_firing,
        code_coeff_bits_per_firing: code_coeff,
        selection_bits_per_firing: sel,
        n_params: feat.n_params,
        l_param_bits: l_param,
        dict_bits,
        code_bits_total: code_total,
        total_bits: total,
        bits_per_token: if feat.n_tokens > 0 {
            total / feat.n_tokens as f64
        } else {
            f64::INFINITY
        },
        residual_achieved: residual,
        distortion_floor: delta2,
        distortion_infeasible: residual > delta2 * 1.02,
    }
}

/// The crossover comparison of a `block` vs a `chart` featurizer at `delta2`.
#[derive(Clone, Debug)]
pub struct Crossover {
    pub block: String,
    pub chart: String,
    pub delta_code_bits_per_firing: f64,
    pub delta_coeff_bits_per_firing: f64,
    pub selection_bits_delta: f64,
    pub selection_asymmetric: bool,
    pub phi_extra_params: i64,
    pub r_per_freed_coord_bits: f64,
    pub l_param_bits: f64,
    pub f_star: f64,
    pub f_star_matched_simple: f64,
    pub chart_wins_at_actual_f: bool,
    pub actual_firings: i64,
}

/// `f*`: the firing count above which the chart's total DL drops below the block's.
///
/// `f* = Φ·L_param / ΔL_per_firing`, with `Φ = P_chart − P_block` the extra decoder
/// scalars and `ΔL_per_firing = (code_b + sel_b) − (code_c + sel_c)` the FULL
/// per-firing bits the chart frees (coefficients **and** selection). Including the
/// selection delta makes this correct even when block and chart differ in `(G, k)`
/// (the `mdl.py` selection-asymmetry fix); when they share `(G, k)` the delta is 0
/// and it reduces to the SNR-independent matched form `Φ/(m_block − m_chart)`.
/// `ΔL_per_firing ≤ 0` (the richer model never pays — a line/control feature)
/// yields `f* = ∞`: the accounting says NO.
pub fn crossover_firings(
    block: &Featurizer,
    chart: &Featurizer,
    delta2: f64,
    l_param_bits: Option<f64>,
) -> Crossover {
    let code_b: f64 = block
        .coded_var
        .iter()
        .map(|&v| scalar_rate_bits(v, delta2))
        .sum();
    let code_c: f64 = chart
        .coded_var
        .iter()
        .map(|&v| scalar_rate_bits(v, delta2))
        .sum();
    let sel_b = selection_bits(block.g_dict, block.k_active);
    let sel_c = selection_bits(chart.g_dict, chart.k_active);
    let dcode_coeff = code_b - code_c;
    let dsel = sel_b - sel_c;
    let dcode = dcode_coeff + dsel;
    let phi = chart.n_params - block.n_params;
    let mb = block.m();
    let r_per_coord = if mb > 0 { code_b / mb as f64 } else { f64::NAN };
    let l_param = l_param_bits.unwrap_or(r_per_coord);
    let f_star = if dcode > 0.0 {
        phi as f64 * l_param / dcode
    } else {
        f64::INFINITY
    };
    let mc = chart.m();
    let f_star_matched = if mb != mc {
        phi as f64 / (mb as f64 - mc as f64)
    } else {
        f64::INFINITY
    };
    Crossover {
        block: block.name.clone(),
        chart: chart.name.clone(),
        delta_code_bits_per_firing: dcode,
        delta_coeff_bits_per_firing: dcode_coeff,
        selection_bits_delta: dsel,
        selection_asymmetric: (block.g_dict, block.k_active) != (chart.g_dict, chart.k_active),
        phi_extra_params: phi,
        r_per_freed_coord_bits: r_per_coord,
        l_param_bits: l_param,
        f_star,
        f_star_matched_simple: f_star_matched,
        chart_wins_at_actual_f: chart.n_firings as f64 >= f_star,
        actual_firings: chart.n_firings,
    }
}

/// A fit's OWN description-length decomposition, in bits, converted from the REML
/// criterion terms (nats). This is the reconciliation side of #2085: the surface
/// bits MUST equal the criterion bits (`v / ln 2`) — same quantity, no parallel
/// accounting. `from_criterion_nats` divides each term by `ln 2`.
#[derive(Clone, Copy, Debug)]
pub struct DescriptionLength {
    pub code_bits: f64,
    pub selection_bits: f64,
    pub dict_bits: f64,
    pub total_bits: f64,
    pub bits_per_token: f64,
}

impl DescriptionLength {
    /// Build from the criterion's term breakdown in NATS: `data_fit` (loss + extra
    /// penalty energy), `sparsity` (assignment selection), and `logdet_occam`
    /// (`½log_det − occam`), over `n_tokens`. Each term is `nats / ln 2` bits.
    pub fn from_criterion_nats(
        data_fit_nats: f64,
        sparsity_nats: f64,
        logdet_occam_nats: f64,
        n_tokens: i64,
    ) -> Self {
        let code = data_fit_nats / LN_2;
        let selection = sparsity_nats / LN_2;
        let dict = logdet_occam_nats / LN_2;
        let total = code + selection + dict;
        Self {
            code_bits: code,
            selection_bits: selection,
            dict_bits: dict,
            total_bits: total,
            bits_per_token: if n_tokens > 0 {
                total / n_tokens as f64
            } else {
                f64::INFINITY
            },
        }
    }

    /// The no-parallel-accounting invariant: the decomposition sums to the total
    /// criterion nll `v` (nats) converted to bits, within `tol` bits. `v` is the
    /// value `SaeManifoldTerm::reml_criterion` returns.
    pub fn reconciles_with_criterion(&self, v_nats: f64, tol_bits: f64) -> bool {
        (self.total_bits - v_nats / LN_2).abs() <= tol_bits
    }
}

#[cfg(test)]
#[path = "description_length_tests.rs"]
mod description_length_tests;

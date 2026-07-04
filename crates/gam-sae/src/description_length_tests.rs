//! Parity tests: the Rust description-length surface must reproduce the
//! hand-verified `Manifold-SAE experiments/mdl_ladder/mdl.py` reference numbers
//! exactly, and the criterion-bits reconciliation invariant must hold.

use super::{
    Crossover, DescriptionLength, Featurizer, ScoreRow, crossover_firings, reverse_water_filling,
    scalar_rate_bits, score, selection_bits,
};
use crate::atom_codes::SparseAtomCodes;

fn feat(
    name: &str,
    kind: &str,
    coded_var: &[f64],
    n_params: i64,
    ev: f64,
    total_var: f64,
    n_tokens: i64,
    n_firings: i64,
    g_dict: i64,
    k_active: i64,
) -> Featurizer {
    Featurizer {
        name: name.to_string(),
        kind: kind.to_string(),
        coded_var: coded_var.to_vec(),
        n_params,
        ev,
        total_var,
        n_tokens,
        n_firings,
        g_dict,
        k_active,
        support_entropy_bits: None,
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn primitives_match_mdl_reference() {
    assert!(close(scalar_rate_bits(1.0, 0.25), 1.160964047, 1e-6));
    assert!(scalar_rate_bits(1.0, 0.0).is_infinite());
    assert!(close(scalar_rate_bits(0.0, 0.5), 0.0, 1e-12));
    // selection bits = log2 C(G, k)
    assert!(close(selection_bits(4096, 1), 12.0, 1e-9));
    assert!(close(selection_bits(32, 4), 15.13410540, 1e-6));
    assert!(close(selection_bits(10, 0), 0.0, 1e-12));
    assert!(close(selection_bits(0, 3), 0.0, 1e-12));
    // k capped at G
    assert!(close(selection_bits(4, 9), selection_bits(4, 4), 1e-12));
}

#[test]
fn reverse_water_filling_matches_mdl_reference() {
    let (rate, per) = reverse_water_filling(&[1.0, 0.5, 0.1], 0.3);
    assert!(close(rate, 2.821928094, 1e-4), "rate {rate}");
    assert!(close(per[0], 1.660964047, 1e-4));
    assert!(close(per[1], 1.160964047, 1e-4));
    assert!(close(per[2], 0.0, 1e-6));
}

#[test]
fn score_matches_mdl_reference() {
    // The programmatic example from mdl_ladder/README.md.
    let block = feat("b2", "block", &[1.10, 0.34], 32, 0.58, 2.55, 35, 35, 1, 1);
    let chart = feat("circle", "chart", &[1.49], 64, 0.584, 2.55, 35, 35, 1, 1);
    let delta2 = chart.residual(); // task-derived floor = best chart residual
    assert!(close(delta2, 1.0608, 1e-4), "delta2 {delta2}");

    let sb: ScoreRow = score(&block, delta2, None);
    assert!(
        close(sb.code_bits_per_firing, 0.7138, 1e-3),
        "code/firing {}",
        sb.code_bits_per_firing
    );
    assert!(close(sb.l_param_bits, 0.3569, 1e-3));
    assert!(close(sb.dict_bits, 11.42, 1e-2));
    assert!(
        close(sb.bits_per_token, 1.04, 1e-2),
        "bpt {}",
        sb.bits_per_token
    );

    let sc = score(&chart, delta2, None);
    assert!(close(sc.code_bits_per_firing, 0.6329, 1e-3));
    assert!(
        close(sc.bits_per_token, 1.7902, 1e-2),
        "bpt {}",
        sc.bits_per_token
    );
    // both feasible at the chart's own residual floor
    assert!(!sc.distortion_infeasible);
}

#[test]
fn crossover_matches_mdl_reference() {
    let block = feat("b2", "block", &[1.10, 0.34], 32, 0.58, 2.55, 35, 35, 1, 1);
    let chart = feat("circle", "chart", &[1.49], 64, 0.584, 2.55, 35, 35, 1, 1);
    let delta2 = chart.residual();
    let xo: Crossover = crossover_firings(&block, &chart, delta2, None);
    assert!(
        close(xo.delta_code_bits_per_firing, 0.0809, 1e-3),
        "dcode {}",
        xo.delta_code_bits_per_firing
    );
    assert_eq!(xo.phi_extra_params, 32);
    assert!(close(xo.l_param_bits, 0.3569, 1e-3));
    assert!(close(xo.f_star, 141.24, 0.5), "f* {}", xo.f_star);
    assert!(!xo.selection_asymmetric);
}

#[test]
fn crossover_charges_selection_delta_when_configs_differ() {
    // The 1d9f843 fix: block/chart with DIFFERENT (G, k) must charge the selection
    // delta into ΔL_per_firing, so f* accounts for the extra selection cost.
    let block = feat("bA", "block", &[1.0, 0.5], 32, 0.5, 2.0, 100, 100, 64, 2);
    let chart = feat("cA", "chart", &[1.2], 80, 0.5, 2.0, 100, 100, 128, 1);
    let xo = crossover_firings(&block, &chart, 0.3, None);
    assert!(xo.selection_asymmetric);
    assert!(
        close(xo.selection_bits_delta, 3.9773, 1e-3),
        "dsel {}",
        xo.selection_bits_delta
    );
    assert!(
        close(xo.delta_code_bits_per_firing, 4.5816, 1e-3),
        "dcode {}",
        xo.delta_code_bits_per_firing
    );
    assert!(close(xo.f_star, 9.25, 0.1), "f* {}", xo.f_star);
    // selection delta really shifted f*: without it (coeff-only) f* would differ.
    assert!((xo.delta_code_bits_per_firing - xo.delta_coeff_bits_per_firing).abs() > 1.0);
}

#[test]
fn crossover_says_no_on_the_control_case() {
    // A line/control "chart" that is no cheaper per firing than the block (or
    // richer without freeing coefficients) must report f* = ∞ — the accounting
    // can say NO. Here the chart codes MORE per firing (dcode < 0).
    let block = feat("line", "block", &[1.0], 16, 0.5, 2.0, 100, 100, 1, 1);
    let chart = feat("curve", "chart", &[1.0, 0.9], 64, 0.5, 2.0, 100, 100, 1, 1);
    let xo = crossover_firings(&block, &chart, 0.3, None);
    assert!(
        xo.f_star.is_infinite(),
        "control chart must never pay: f*={}",
        xo.f_star
    );
    assert!(!xo.chart_wins_at_actual_f);
}

#[test]
fn criterion_bits_reconcile_no_parallel_accounting() {
    use std::f64::consts::LN_2;
    // The REML criterion v = data_fit + sparsity + (½log_det − occam) in nats; the
    // surface bits are exactly v/ln2, split term-for-term (no drift).
    let (data_fit, sparsity, logdet_occam, n) = (128.0, 7.5, 40.0, 50_000);
    let dl: DescriptionLength =
        DescriptionLength::from_criterion_nats(data_fit, sparsity, logdet_occam, n);
    let v = data_fit + sparsity + logdet_occam;
    assert!(close(dl.code_bits, data_fit / LN_2, 1e-9));
    assert!(close(dl.selection_bits, sparsity / LN_2, 1e-9));
    assert!(close(dl.dict_bits, logdet_occam / LN_2, 1e-9));
    assert!(close(dl.total_bits, v / LN_2, 1e-9));
    assert!(close(dl.bits_per_token, v / LN_2 / n as f64, 1e-12));
    // The invariant a REML-fitted atom's surface must satisfy.
    assert!(dl.reconciles_with_criterion(v, 1e-9));
    assert!(!dl.reconciles_with_criterion(v + 10.0 * LN_2, 1.0)); // a 10-bit drift is caught
}

/// A strongly-structured TILING dictionary: each token activates a contiguous
/// run of adjacent atoms, so which atoms fire is highly predictable (adjacent
/// atoms co-fire). Two accounting facts the reviewer required must hold:
///
///  1. the empirical support entropy `H(S)` (Chow–Liu tree bits) is FAR below the
///     combinatorial worst case `log₂ C(G, k̄)` — the uniform price overpays this
///     dictionary, so charging it would let the MDL comparison argue with itself;
///  2. charging the tiling baseline `H(S)` instead of the combinatorial price
///     lowers its selection cost, which SHRINKS a richer chart's reported MDL gap
///     over it — the direction the math says (the overpay was inflating the win).
fn tiling_codes(n: usize, g: usize, run: usize, seed: u64) -> SparseAtomCodes {
    // A tiling dictionary of `g/run` disjoint tiles, each a contiguous block of
    // `run` adjacent atoms that fires as a UNIT; every token activates exactly one
    // tile (chosen deterministically). Within a tile the atoms are perfectly
    // co-firing, so the pairwise (Chow–Liu tree) model captures the support
    // structure almost exactly — the predictable adjacent-atom co-firing the
    // reviewer's correction is about — while the uniform combinatorial price still
    // charges `log₂ C(G, run)` as if every `run`-subset were possible.
    let n_tiles = g / run;
    let mut codes = SparseAtomCodes::empty(n, g);
    let mut state = seed | 1;
    for row in 0..n {
        // A cheap LCG step just to spread the tile choices deterministically.
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let tile = (state >> 33) as usize % n_tiles;
        for off in 0..run {
            codes.row_mut(row).assign(tile * run + off, 1.0);
        }
    }
    codes
}

#[test]
fn tiling_support_entropy_undercuts_combinatorial_and_shrinks_gap() {
    let (n, g, run) = (4000usize, 48usize, 3usize);
    let codes = tiling_codes(n, g, run, 0xC0FFEE);
    let se = codes.support_entropy();

    // The run is exactly `run` atoms wide on every row.
    assert!(
        (se.mean_support - run as f64).abs() < 1e-9,
        "mean support {} should equal the run width {run}",
        se.mean_support
    );
    // Chow–Liu bounds: 0 ≤ H(S) ≤ independent ≤ (not necessarily) combinatorial,
    // but for a PREDICTABLE tiling H(S) must be WELL below the combinatorial price.
    assert!(se.tree_bits <= se.independent_bits + 1e-9);
    assert!(
        se.tree_bits < 0.6 * se.combinatorial_bits,
        "tiling H(S)={} must be far below combinatorial log2 C(G,k̄)={}",
        se.tree_bits,
        se.combinatorial_bits
    );

    // The reviewer's comparison: a TILING SAE baseline (large G, k>1 co-firing —
    // the block, which the combinatorial price overpays) versus a single-atom
    // manifold CHART that reads one coordinate (g_dict = 1, k = 1 → ~zero
    // selection cost, no redundant support to price). Only the tiling block
    // carries the empirical H(S); the chart's selection price is 0 either way.
    let block = feat(
        "tiling-block", "block", &[1.0, 0.5, 0.5], 64, 0.5, 3.0, n as i64, n as i64, g as i64, run as i64,
    );
    let chart = feat(
        "manifold-chart", "chart", &[1.2], 160, 0.55, 3.0, n as i64, n as i64, 1, 1,
    );
    let delta2 = 0.4;

    // Combinatorial accounting (no support entropy attached): the block pays the
    // full log₂ C(G, k) selection price.
    let xo_comb = crossover_firings(&block, &chart, delta2, None);

    // Entropy-corrected accounting: charge the tiling block its true H(S).
    let block_h = block.clone().with_support_entropy(&codes);
    let xo_h = crossover_firings(&block_h, &chart, delta2, None);

    // The reviewer's worst-case line is preserved unchanged.
    assert!(
        (xo_h.selection_bits_delta_combinatorial - xo_comb.selection_bits_delta).abs() < 1e-9,
        "combinatorial delta must be reported alongside and match the uncorrected run"
    );

    // The correction lowered the block's selection cost, so the block frees FEWER
    // selection bits per firing to the chart (`Δsel` drops): the chart's per-firing
    // advantage shrinks.
    assert!(
        xo_h.selection_bits_delta < xo_comb.selection_bits_delta - 1e-6,
        "entropy correction must reduce the block→chart selection delta: {} !< {}",
        xo_h.selection_bits_delta,
        xo_comb.selection_bits_delta
    );

    // Direction of the reported gap: `f*` is the firing count above which the
    // chart's total DL beats the block's. Charging the tiling baseline its true
    // (cheaper) selection price makes the chart pay for its extra decoder params
    // over more firings — `f*` RISES (the chart wins later), i.e. the previously
    // inflated MDL gap shrinks. Both configs free coefficients here so f* is finite.
    assert!(xo_comb.f_star.is_finite() && xo_h.f_star.is_finite());
    assert!(
        xo_h.f_star > xo_comb.f_star + 1e-6,
        "entropy correction must shrink the chart's advantage (f* rises): {} !> {}",
        xo_h.f_star,
        xo_comb.f_star
    );
}

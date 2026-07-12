//! Parity tests: the Rust description-length surface must reproduce the
//! hand-verified `Manifold-SAE experiments/mdl_ladder/mdl.py` reference numbers
//! exactly, and the criterion-bits reconciliation invariant must hold.

use super::{
    BirthMdlPrescreen, Crossover, DescriptionLength, Featurizer, ScoreRow,
    bar_birth_threshold_nats, bar_supports_birth, circle_chart_columns, circle_coding_gain_bits,
    circle_shape_const_bits, crossover_firings, curved_coding_gain_bits,
    evidence_per_log_persistence, kappa_coding_gain_detector, manifold_fit_description_length,
    matched_dl, matched_dl_delta, predicted_birth_dl_bits, reverse_water_filling, scalar_rate_bits,
    score, se_resolution_bits, selection_bits, uniform_unit_range_sd,
    weighted_reverse_water_filling,
};
use crate::atom_codes::SparseAtomCodes;

/// A small deterministic support: `n` tokens over `g` atoms, atom `k` firing on
/// every `(k+1)`-th token, so supports vary in cardinality (exercising the
/// empirical support-entropy selection code).
fn planted_codes(n: usize, g: usize) -> SparseAtomCodes {
    let mut codes = SparseAtomCodes::empty(n, g);
    for row in 0..n {
        for atom in 0..g {
            if row % (atom + 1) == 0 {
                codes.row_mut(row).assign(atom, 1.0);
            }
        }
    }
    codes
}

#[test]
fn manifold_fit_dl_decomposes_and_sums_to_total() {
    // N=40 tokens, G=4 atoms, d_k=1 coord each, a 2-coordinate variance spectrum
    // coded to total distortion 0.2, 96 decoder scalars at the distortion-matched
    // precision (l_param defaults to the per-coordinate code rate).
    let codes = planted_codes(40, 4);
    let coord_variances = [1.0_f64, 0.5];
    let delta2 = 0.2;
    let atom_dims = [1.0_f64, 1.0, 1.0, 1.0];
    let dl = manifold_fit_description_length(
        &codes,
        &coord_variances,
        delta2,
        &atom_dims,
        0.9,
        96,
        None,
    );

    // Per-coordinate rate is the reverse-water-filling mean of the actual spectrum.
    let (coord_total, _) = reverse_water_filling(&coord_variances, delta2);
    let rate = coord_total / coord_variances.len() as f64;
    assert!((dl.coordinate_rate_bits - rate).abs() < 1e-12);
    assert!(
        (dl.l_param_bits - rate).abs() < 1e-12,
        "default l_param = code rate"
    );

    // Selection = empirical support entropy H(S); code = every fired coded scalar.
    let support = codes.support_entropy();
    assert!((dl.selection_bits_per_token - support.tree_bits).abs() < 1e-12);
    let coded_scalars: f64 = codes
        .iter()
        .map(|c| c.active_mask.count_ones() as f64)
        .sum();
    assert!((dl.code_bits - coded_scalars * rate).abs() < 1e-9);

    // Corpus totals and per-token accounting reconcile with the parts.
    assert!((dl.selection_bits - 40.0 * dl.selection_bits_per_token).abs() < 1e-9);
    assert!((dl.dict_bits - 96.0 * rate).abs() < 1e-12);
    let total = dl.code_bits + dl.selection_bits + dl.dict_bits;
    assert!(
        (dl.total_bits - total).abs() < 1e-9,
        "ledgers must sum to the total"
    );
    assert!((dl.bits_per_token - dl.total_bits / 40.0).abs() < 1e-9);
    assert!(
        (dl.dict_bits_per_token - dl.dict_bits / 40.0).abs() < 1e-9,
        "dictionary bits are amortised across the corpus"
    );
}

#[test]
fn manifold_fit_dl_code_rate_rises_as_distortion_tightens() {
    // The honest rate–distortion trade: a finer distortion floor (smaller delta2)
    // costs MORE per-coordinate code bits, from the actual variance spectrum.
    let codes = planted_codes(30, 4);
    let coord_variances = [1.0_f64, 0.6, 0.3];
    let atom_dims = [1.0_f64, 1.0, 1.0, 1.0];
    let coarse =
        manifold_fit_description_length(&codes, &coord_variances, 0.6, &atom_dims, 0.5, 64, None);
    let fine =
        manifold_fit_description_length(&codes, &coord_variances, 0.1, &atom_dims, 0.95, 64, None);
    assert!(
        fine.coordinate_rate_bits > coarse.coordinate_rate_bits,
        "a tighter distortion must cost more per-coordinate bits: {} !> {}",
        fine.coordinate_rate_bits,
        coarse.coordinate_rate_bits
    );
    assert!(fine.code_bits_per_token > coarse.code_bits_per_token);
}

#[test]
fn manifold_fit_dl_tight_distortion_stays_finite() {
    // A tight (but positive) distortion drives the rate large; as long as it stays
    // above zero the reported bits are large yet finite (never +∞).
    let codes = planted_codes(10, 4);
    let coord_variances = [1.0_f64, 0.5];
    let atom_dims = [1.0_f64, 1.0, 1.0, 1.0];
    let dl = manifold_fit_description_length(
        &codes,
        &coord_variances,
        1.0e-6,
        &atom_dims,
        0.999,
        8,
        None,
    );
    assert!(dl.bits_per_token.is_finite());
    assert!(dl.coordinate_rate_bits.is_finite() && dl.coordinate_rate_bits > 0.0);
}

#[test]
fn manifold_fit_dl_explicit_l_param_overrides_default() {
    // Passing fp16 precision (16 bits/scalar) must be used verbatim for the
    // dictionary charge instead of the distortion-matched default.
    let codes = planted_codes(20, 4);
    let coord_variances = [1.0_f64, 0.5];
    let atom_dims = [1.0_f64, 1.0, 1.0, 1.0];
    let dl = manifold_fit_description_length(
        &codes,
        &coord_variances,
        0.2,
        &atom_dims,
        0.8,
        50,
        Some(16.0),
    );
    assert!((dl.l_param_bits - 16.0).abs() < 1e-12);
    assert!((dl.dict_bits - 50.0 * 16.0).abs() < 1e-9);
}

#[test]
fn se_resolution_bits_is_the_uniform_quantization_cost() {
    // The closed form ½·log₂(1/(12·SE²)): a coordinate known to SE = 0.01 on a unit
    // range costs exactly −½·log₂(12·0.01²) bits.
    let se = 0.01;
    let got = se_resolution_bits(se);
    let expected = -0.5 * (12.0 * se * se).log2();
    assert!(
        (got - expected).abs() < 1e-12,
        "got {got} expected {expected}"
    );
    assert!(
        got > 0.0,
        "a well-localized coordinate must carry positive bits"
    );
    // At the uniform-prior ceiling SE = 1/√12 the cost is exactly 0 (no info beyond
    // the U(0,1) prior); above it, still 0 (floored).
    let ceil = uniform_unit_range_sd();
    assert!(
        se_resolution_bits(ceil).abs() < 1e-12,
        "ceiling SE must cost 0 bits"
    );
    assert_eq!(
        se_resolution_bits(2.0 * ceil),
        0.0,
        "above-ceiling SE costs 0 bits"
    );
    // Halving SE adds exactly one bit (a factor-2 finer resolution).
    let d = se_resolution_bits(se / 2.0) - se_resolution_bits(se);
    assert!(
        (d - 1.0).abs() < 1e-12,
        "halving SE must add exactly 1 bit, got {d}"
    );
}

#[test]
fn matched_dl_planted_circle_gives_closed_form_bit_count() {
    // A planted circle chart of known harmonic order H fired f times, each firing at
    // a known coordinate SE, in ambient p, at l_param bits/scalar. The matched
    // description length must equal the hand-computed closed form exactly:
    //   total = (2H+1)·p·l_param  +  f · ½log₂(1/(12·SE²)).
    let h = 3usize; // harmonic order
    let p = 64i64; // ambient dim
    let l_param = 4.0; // bits per stored scalar
    let se = 0.02; // per-firing coordinate SE (σ/(2π‖z‖))
    let f = 250usize; // firings
    let columns = circle_chart_columns(h);
    assert_eq!(columns, 7, "2H+1 = 7 for H=3");

    let ses = vec![se; f];
    let ev = 0.4;
    // A circle chart transmits ONE phase coordinate per firing.
    let dl = matched_dl(columns, 1, p, l_param, &ses, ev);

    let expected_param = 7.0 * 64.0 * 4.0;
    let expected_coding = f as f64 * (-0.5 * (12.0 * se * se).log2());
    let expected_total = expected_param + expected_coding;
    assert!(
        (dl.param_bits - expected_param).abs() < 1e-9,
        "param bits {} vs {expected_param}",
        dl.param_bits
    );
    assert!(
        (dl.coding_bits - expected_coding).abs() < 1e-6,
        "coding bits {} vs {expected_coding}",
        dl.coding_bits
    );
    assert!(
        (dl.total_dl_bits - expected_total).abs() < 1e-6,
        "total DL bits {} vs {expected_total}",
        dl.total_dl_bits
    );
    assert_eq!(dl.n_firings, f as i64);
    assert!((dl.dl_per_ev - expected_total / ev).abs() < 1e-6);

    // Matched-DL delta vs the flat / line atom (1 column, 1 amplitude per firing at
    // the SAME SE): both arms transmit ONE scalar per firing at the same SE so the
    // coding bits cancel, and the curved chart pays 2H extra columns of parameter
    // charge — at large p the flat atom is the shorter code here (delta < 0 — the
    // honest "curvature doesn't pay at these firings" verdict). This is the
    // primitive's equal-SE behavior; the real per-arm phase-vs-amplitude SE
    // distinction (the 2π factor) is pinned in
    // `matched_dl_per_arm_phase_vs_amplitude_rate_removes_pro_chart_bias`.
    let flat = matched_dl(1, 1, p, l_param, &ses, ev);
    let delta = matched_dl_delta(&flat, &dl);
    let expected_delta = flat.total_dl_bits - dl.total_dl_bits;
    assert!((delta - expected_delta).abs() < 1e-9);
    assert!(
        (delta - (1.0 - 7.0) * 64.0 * 4.0).abs() < 1e-6,
        "delta must be the pure param-column difference (coding bits cancel): {delta}"
    );
    assert!(
        delta < 0.0,
        "flat cheaper than a 7-column chart at equal firings"
    );

    // The code-economy axis: a b=4 flat BLOCK transmits 4 coefficients per firing
    // where the chart transmits 1, so the block pays 3·Σ bits(SE) extra coding
    // bits — enough firings and the chart wins on code economy alone even against
    // a cheaper dictionary.
    let block = matched_dl(4, 4, p, l_param, &ses, ev);
    let per_firing_bits = -0.5 * (12.0 * se * se).log2();
    assert!(
        (block.coding_bits - 4.0 * f as f64 * per_firing_bits).abs() < 1e-6,
        "block codes 4 scalars per firing: {}",
        block.coding_bits
    );
    let economy_delta = matched_dl_delta(&block, &dl);
    let expected_economy = (4.0 - 7.0) * 64.0 * 4.0 + 3.0 * f as f64 * per_firing_bits;
    assert!(
        (economy_delta - expected_economy).abs() < 1e-6,
        "delta = param-column difference + per-firing economy: {economy_delta} vs {expected_economy}"
    );
    assert!(
        economy_delta > 0.0,
        "at 250 firings the chart's per-firing economy beats its extra columns"
    );
}

#[test]
fn matched_dl_per_arm_phase_vs_amplitude_rate_removes_pro_chart_bias() {
    use std::f64::consts::TAU;
    // The S5 fix: a circle chart codes ONE phase per firing at the phase SE
    // σ̂/(2π‖z‖); a flat b-block codes b AMPLITUDES per firing at the amplitude SE
    // σ̂/‖z‖ = 2π·SE_phase. Pricing the flat amplitudes at the finer PHASE SE (the
    // old shared-list arithmetic) overcharged the flat arm by log₂(2π) bits per
    // coded scalar — a pro-chart bias. Pin the corrected closed form and the SIGN
    // of the correction relative to the biased delta.
    let p = 64i64;
    let l_param = 3.0;
    let b = 4i64; // flat block coordinates per firing
    let f = 200usize; // firings
    let ev = 0.5;
    let sigma = 0.3; // radial scatter σ̂
    let radius = 2.0; // firing radius ‖z‖ (constant ⇒ a clean closed form)
    let se_phase = sigma / (TAU * radius); // σ̂/(2π‖z‖)
    let se_amp = sigma / radius; // σ̂/‖z‖ = 2π·SE_phase
    assert!((se_amp - TAU * se_phase).abs() < 1e-12);

    let phase_ses = vec![se_phase; f];
    let amp_ses = vec![se_amp; f];

    // Corrected arms: flat codes its b amplitudes at SE_amp, chart its 1 phase at
    // SE_phase — each at its OWN resolution.
    let flat = matched_dl(b, b, p, l_param, &amp_ses, ev);
    let chart = matched_dl(1, 1, p, l_param, &phase_ses, ev);

    let phase_bits = se_resolution_bits(se_phase);
    let amp_bits = se_resolution_bits(se_amp);
    // Closed form: coding = coords_per_firing · f · se_resolution_bits(SE).
    assert!(
        (flat.coding_bits - b as f64 * f as f64 * amp_bits).abs() < 1e-6,
        "flat codes b amplitudes at the amplitude SE: {}",
        flat.coding_bits
    );
    assert!(
        (chart.coding_bits - f as f64 * phase_bits).abs() < 1e-6,
        "chart codes 1 phase at the phase SE: {}",
        chart.coding_bits
    );
    // The per-coordinate phase↔amplitude gap is EXACTLY log₂(2π): the phase, read
    // over the circumference 2π‖z‖, resolves 2π finer than an amplitude over ‖z‖.
    assert!(
        (phase_bits - amp_bits - TAU.log2()).abs() < 1e-9,
        "phase SE is 2π finer than amplitude SE ⇒ log₂(2π) more bits per coordinate"
    );

    let corrected_delta = matched_dl_delta(&flat, &chart); // flat − chart, bits

    // The OLD biased arithmetic priced the flat amplitudes at the PHASE SE too.
    let flat_biased = matched_dl(b, b, p, l_param, &phase_ses, ev);
    let biased_delta = matched_dl_delta(&flat_biased, &chart);

    // Sign of the correction: matched_dl_delta = flat − chart (positive ⇒ chart is
    // the shorter code). Overcharging the flat amplitudes inflated flat.total, which
    // inflated flat − chart — i.e. OVERSTATED the chart's advantage. Coding each
    // amplitude at its own (coarser) SE removes exactly b·f·log₂(2π) bits from the
    // flat arm, so the corrected delta is LOWER by that bias.
    let removed_bias = b as f64 * f as f64 * TAU.log2();
    assert!(
        (corrected_delta - (biased_delta - removed_bias)).abs() < 1e-6,
        "corrected delta must drop by the removed bias: {corrected_delta} vs {}",
        biased_delta - removed_bias
    );
    assert!(
        corrected_delta < biased_delta,
        "the correction removes a PRO-CHART bias, so flat − chart must DROP: {corrected_delta} !< {biased_delta}"
    );
    // Parameter-column ledger is untouched by the coding-rate correction.
    assert!((flat.param_bits - b as f64 * p as f64 * l_param).abs() < 1e-9);
    assert!((chart.param_bits - 1.0 * p as f64 * l_param).abs() < 1e-9);
}

#[test]
fn circle_gain_matches_closed_form() {
    use std::f64::consts::PI;
    let a = 1.7;
    let delta = 0.05;
    let got = circle_coding_gain_bits(a, delta);
    let expected = 0.5 * (3.0 * a * a / (PI * PI * delta * delta)).log2();
    assert!(
        (got - expected).abs() < 1e-12,
        "got {got} expected {expected}"
    );
}

#[test]
fn general_gain_with_circle_shape_const_equals_circle_gain() {
    // D − d = 1 for the circle; feeding its shape constant into the general
    // formula must reproduce the exact circle gain.
    let a = 2.3;
    let delta = 0.02;
    let c = circle_shape_const_bits(a);
    let general = curved_coding_gain_bits(2.0, 1.0, delta, c);
    let circle = circle_coding_gain_bits(a, delta);
    assert!(
        (general - circle).abs() < 1e-12,
        "general {general} vs circle {circle}"
    );
}

#[test]
fn codimension_dividend_is_half_log_per_direction() {
    let delta = 0.1;
    let one = curved_coding_gain_bits(3.0, 2.0, delta, 0.0);
    let two = curved_coding_gain_bits(4.0, 2.0, delta, 0.0);
    let per_dir = 0.5 * (1.0 / (delta * delta)).log2();
    assert!((one - per_dir).abs() < 1e-12);
    assert!((two - 2.0 * per_dir).abs() < 1e-12);
}

#[test]
fn no_gain_without_codimension() {
    assert_eq!(curved_coding_gain_bits(3.0, 3.0, 0.1, 5.0), 0.0);
    assert_eq!(curved_coding_gain_bits(2.0, 3.0, 0.1, 5.0), 0.0);
    assert_eq!(circle_coding_gain_bits(0.0, 0.1), 0.0);
}

#[test]
fn bar_threshold_matches_formula_and_gates_correctly() {
    let delta_d_eff = 4.0;
    let n_eff = std::f64::consts::E.powf(3.0); // ln = 3
    let codim = 2.0;
    let thr = bar_birth_threshold_nats(delta_d_eff, n_eff, codim);
    // ½·4·3 / (n_eff·2) = 3/n_eff.
    let expected = 3.0 / n_eff;
    assert!(
        (thr - expected).abs() < 1e-12,
        "thr {thr} expected {expected}"
    );

    let birth = 1.0;
    let death_pass = (thr * 1.01).exp();
    let death_fail = (thr * 0.99).exp();
    assert!(bar_supports_birth(
        birth,
        death_pass,
        delta_d_eff,
        n_eff,
        codim
    ));
    assert!(!bar_supports_birth(
        birth,
        death_fail,
        delta_d_eff,
        n_eff,
        codim
    ));
}

#[test]
fn exchange_rate_is_neff_times_codim() {
    assert_eq!(evidence_per_log_persistence(128.0, 3.0), 384.0);
}

#[test]
fn kappa_detector_zero_only_at_gaussian_anchor() {
    assert_eq!(kappa_coding_gain_detector(2.0), 0.0);
    assert!(kappa_coding_gain_detector(1.0) > 0.0); // super-Gaussian gate
    assert!(kappa_coding_gain_detector(1.5) > 0.0); // sub-Gaussian circle
}

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
    assert!(close(scalar_rate_bits(1.0, 0.25), 1.0, 1e-12));
    assert!(close(scalar_rate_bits(1.0, 1.0), 0.0, 1e-12));
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
fn reverse_water_filling_spends_one_total_distortion_budget() {
    let (rate, per) = reverse_water_filling(&[1.0, 1.0], 0.5);
    assert!(close(rate, 2.0, 1e-12), "rate {rate}");
    assert!(close(per[0], 1.0, 1e-12));
    assert!(close(per[1], 1.0, 1e-12));
}

#[test]
fn weighted_water_filling_solves_shared_level_exactly() {
    let rates =
        weighted_reverse_water_filling(&[(0.5, vec![1.0, 0.25]), (1.0, vec![0.5])], 0.375).unwrap();
    // D(theta)=0.5*min(1,theta)+0.5*min(.25,theta)+min(.5,theta).
    // At theta=.25 this is .125+.125+.25=.5, so D=.375 lies in the first
    // segment with total active weight 2 and theta=.1875.
    let theta = 0.1875_f64;
    let expected_first = 0.5 * (scalar_rate_bits(1.0, theta) + scalar_rate_bits(0.25, theta));
    let expected_second = scalar_rate_bits(0.5, theta);
    assert!(close(rates[0], expected_first, 1.0e-12));
    assert!(close(rates[1], expected_second, 1.0e-12));
}

#[test]
fn score_uses_reverse_water_filling_for_total_distortion() {
    let block = feat("b2", "block", &[1.10, 0.34], 32, 0.58, 2.55, 35, 35, 1, 1);
    let chart = feat("circle", "chart", &[1.49], 64, 0.584, 2.55, 35, 35, 1, 1);
    let delta2 = chart.residual(); // task-derived floor = best chart residual
    assert!(close(delta2, 1.0608, 1e-4), "delta2 {delta2}");

    let sb: ScoreRow = score(&block, delta2, None);
    let (expected_block_rate, _) = reverse_water_filling(&block.coded_var, delta2);
    assert!(close(
        sb.code_coeff_bits_per_firing,
        expected_block_rate,
        1e-12
    ));
    assert!(close(sb.l_param_bits, expected_block_rate / 2.0, 1e-12));
    assert!(close(sb.dict_bits, 32.0 * sb.l_param_bits, 1e-12));

    let sc = score(&chart, delta2, None);
    let (expected_chart_rate, _) = reverse_water_filling(&chart.coded_var, delta2);
    assert!(close(
        sc.code_coeff_bits_per_firing,
        expected_chart_rate,
        1e-12
    ));
    // both feasible at the chart's own residual floor
    assert!(!sc.distortion_infeasible);
}

#[test]
fn crossover_uses_the_same_total_distortion_currency_as_score() {
    let block = feat("b2", "block", &[1.10, 0.34], 32, 0.58, 2.55, 35, 35, 1, 1);
    let chart = feat("circle", "chart", &[1.49], 64, 0.584, 2.55, 35, 35, 1, 1);
    let delta2 = chart.residual();
    let xo: Crossover = crossover_firings(&block, &chart, delta2, None);
    let block_score = score(&block, delta2, None);
    let chart_score = score(&chart, delta2, None);
    assert!(close(
        xo.delta_code_bits_per_firing,
        block_score.code_bits_per_firing - chart_score.code_bits_per_firing,
        1e-12,
    ));
    assert_eq!(xo.phi_extra_params, 32);
    assert!(xo.f_star.is_finite() && xo.f_star > 0.0);
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
    assert!(close(
        xo.delta_code_bits_per_firing,
        xo.delta_coeff_bits_per_firing + xo.selection_bits_delta,
        1e-12,
    ));
    assert!(xo.f_star.is_finite() && xo.f_star > 0.0);
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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        "tiling-block",
        "block",
        &[1.0, 0.5, 0.5],
        64,
        0.5,
        3.0,
        n as i64,
        n as i64,
        g as i64,
        run as i64,
    );
    let chart = feat(
        "manifold-chart",
        "chart",
        &[1.2],
        160,
        0.55,
        3.0,
        n as i64,
        n as i64,
        1,
        1,
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

/// #2233 closed-form birth pre-screen: hand-computed crossover on a planted
/// spectrum. A circle (span 2, no code savings, support-only win) pays; a torus
/// (span 4, rich basis) at a tiny activation rate does NOT.
#[test]
fn birth_prescreen_matches_hand_computed_crossover() {
    // Shared config: N=1000 tokens, P=8 channels, G=1024 atoms, L0=32 active.
    // log₂(N) = ln(1000)/ln(2) and log₂(G/L0) = log₂(32) = 5 exactly.
    let log2_n = 1000.0_f64.log2();
    let log2_g_over_l0 = (1024.0_f64 / 32.0).log2();
    assert!((log2_g_over_l0 - 5.0).abs() < 1e-12);

    // --- Circle: span ŝ=2, d=1, m=3. Code term (ŝ−d−1)=0 ⇒ support-only win. ---
    // λ̂=3, δ=1 (scalar rate ½log₂(3), unused here since the code coefficient is 0).
    let circle = BirthMdlPrescreen {
        rho: 0.1,
        span: 2.0,
        intrinsic_dim: 1,
        basis_size: 3,
        signal_var: 3.0,
        noise_floor: 1.0,
        n_tokens: 1000.0,
        p_out: 8,
        g_dict: 1024,
        l0: 32.0,
    };
    // saving = ρN·[0 + (s−1)·log₂(G/L0)] = 0.1·1000·(1·5) = 500.
    // surcharge = (m−s)·P·½·log₂(N) = (3−2)·8·0.5·log₂(1000).
    let expected_circle = 0.1 * 1000.0 * 5.0 - (3.0 - 2.0) * 8.0 * 0.5 * log2_n;
    let got_circle = predicted_birth_dl_bits(&circle);
    assert!(
        (got_circle - expected_circle).abs() < 1e-9,
        "circle pre-screen bits {got_circle} != hand value {expected_circle}"
    );
    assert!(got_circle > 0.0, "a firing circle must pay: {got_circle}");

    // --- Torus at tiny ρ: span ŝ=4, d=2, m=25. Rich basis, negligible firing. ---
    let torus = BirthMdlPrescreen {
        rho: 0.001,
        span: 4.0,
        intrinsic_dim: 2,
        basis_size: 25,
        signal_var: 3.0,
        noise_floor: 1.0,
        n_tokens: 1000.0,
        p_out: 8,
        g_dict: 1024,
        l0: 32.0,
    };
    // code coeff (s−d−1)=1, scalar rate = ½log₂(3); support=(4−1)·5=15.
    let scalar_rate = 0.5 * 3.0_f64.log2();
    let expected_torus = 0.001 * 1000.0 * (scalar_rate + 15.0) - (25.0 - 4.0) * 8.0 * 0.5 * log2_n;
    let got_torus = predicted_birth_dl_bits(&torus);
    assert!(
        (got_torus - expected_torus).abs() < 1e-9,
        "torus pre-screen bits {got_torus} != hand value {expected_torus}"
    );
    assert!(
        got_torus < 0.0,
        "a tiny-ρ rich torus must not pay (deferred): {got_torus}"
    );

    assert!((scalar_rate_bits(3.0, 1.0) - scalar_rate).abs() < 1e-12);
}

/// A planted `s = d+1` kind (circle `d=1`, sphere `d=2`): `s` orthogonal
/// equal-energy signal columns (distinct-frequency cosines are discretely
/// orthogonal on evenly spaced angles, so each carries variance `r²/2` and the
/// curved atom's contribution SVD yields exactly `s` equal singular values),
/// plus deterministic isotropic noise on every ambient channel. `recon` is the
/// denoised signal, shared by the flat and curved featurizers, so their Eq-4
/// residual and code terms are identical by construction and the verdict is
/// driven purely by support (flat spends `s` active slots, curved spends 1) vs
/// the dictionary surcharge (`m·P` vs `s·P`) — exactly the `s=d+1` support-only
/// crossover the pre-screen prices.
fn planted_sd1_signal(n: usize, p: usize, d: usize, radius: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let s = d + 1;
    let mut signal = vec![vec![0.0_f64; p]; n];
    let mut noisy = vec![vec![0.0_f64; p]; n];
    let noise_sd = 0.1_f64;
    for (i, (sig_row, noisy_row)) in signal.iter_mut().zip(noisy.iter_mut()).enumerate() {
        for (j, sig) in sig_row.iter_mut().enumerate().take(s) {
            let freq = (j + 1) as f64;
            *sig = radius * (std::f64::consts::TAU * freq * i as f64 / n as f64).cos();
        }
        for (j, noisy_val) in noisy_row.iter_mut().enumerate() {
            // Deterministic, channel- and row-varying noise (incommensurate with
            // the signal frequencies) so the residual has full-rank isotropic mass.
            let noise = noise_sd * (0.7 * i as f64 + 1.9 * j as f64 + 0.3).sin();
            *noisy_val = sig_row[j] + noise;
        }
    }
    (noisy, signal)
}

/// The signed per-token Eq-4 bits advantage of the CURVED single-atom featurizer
/// over the FLAT `s`-latent alternative on a planted `s=d+1` kind (positive ⇒
/// curved is cheaper ⇒ the birth wins), computed through the production
/// [`crate::eq4_description_length::eq4_fixed_distortion_description_length`]
/// path — and the closed-form pre-screen's prediction on the SAME planted
/// quantities. Both should carry the same sign: that agreement is the pre-screen's
/// contract.
fn eq4_curved_advantage_and_prescreen(
    d: usize,
    n: usize,
    p: usize,
    g_dict: usize,
    basis_m: usize,
) -> (f64, f64) {
    use crate::eq4_description_length::eq4_fixed_distortion_description_length;
    use ndarray::Array2;

    let s = d + 1;
    let radius = 2.0_f64;
    let target = 0.9_f64;
    let (noisy, signal) = planted_sd1_signal(n, p, d, radius);
    let test_x = Array2::from_shape_fn((n, p), |(i, j)| noisy[i][j]);
    // Both featurizers reconstruct the same denoised signal: circle/sphere sit in
    // their linear span, so the curved atom and the flat latents recover it
    // identically, isolating the support-vs-dictionary crossover.
    let recon = Array2::from_shape_fn((n, p), |(i, j)| signal[i][j]);
    let signal_mat = Array2::from_shape_fn((n, p), |(i, j)| signal[i][j]);

    // FLAT: s atoms (columns 0..s), each firing every row, one coded coordinate.
    let mut flat_gate = Array2::zeros((n, g_dict));
    let mut flat_dims = vec![0_i64; g_dict];
    for atom in 0..s {
        for row in 0..n {
            flat_gate[[row, atom]] = 1.0;
        }
        flat_dims[atom] = 1;
    }
    let flat = {
        let signal_mat = signal_mat.clone();
        eq4_fixed_distortion_description_length(
            test_x.view(),
            recon.view(),
            flat_gate.view(),
            &flat_dims,
            (s * p) as i64,
            &[target],
            None,
            move |atom, take| {
                // Flat atom `atom` reconstructs only its own signal channel.
                let mut out = Array2::zeros((take.len(), p));
                for (out_row, &src) in take.iter().enumerate() {
                    out[[out_row, atom]] = signal_mat[[src, atom]];
                }
                Ok(out)
            },
        )
        .expect("flat Eq-4 scoring succeeds")
    };

    // CURVED: one atom (column 0), firing every row, d+1 coded coordinates.
    let mut curved_gate = Array2::zeros((n, g_dict));
    for row in 0..n {
        curved_gate[[row, 0]] = 1.0;
    }
    let mut curved_dims = vec![0_i64; g_dict];
    curved_dims[0] = s as i64;
    let curved = {
        let signal_mat = signal_mat.clone();
        eq4_fixed_distortion_description_length(
            test_x.view(),
            recon.view(),
            curved_gate.view(),
            &curved_dims,
            (basis_m * p) as i64,
            &[target],
            None,
            move |_atom, take| {
                // The curved atom reconstructs the whole s-dimensional signal.
                let mut out = Array2::zeros((take.len(), p));
                for (out_row, &src) in take.iter().enumerate() {
                    for col in 0..s {
                        out[[out_row, col]] = signal_mat[[src, col]];
                    }
                }
                Ok(out)
            },
        )
        .expect("curved Eq-4 scoring succeeds")
    };

    let eq4_advantage = flat.per_target[0].bits - curved.per_target[0].bits;

    let predicted = predicted_birth_dl_bits(&BirthMdlPrescreen {
        rho: 1.0,
        span: s as f64,
        intrinsic_dim: d,
        basis_size: basis_m,
        signal_var: radius * radius / 2.0,
        noise_floor: 0.01,
        n_tokens: n as f64,
        p_out: p,
        g_dict,
        l0: s as f64,
    });

    (eq4_advantage, predicted)
}

/// #2233 agreement contract: the closed-form birth pre-screen's win/lose VERDICT
/// (the sign of its predicted ΔMDL) must match the full Eq-4 fixed-distortion
/// bits computation run through the production scorer on the same planted
/// `s=d+1` data — for a circle (`d=1`) and a sphere (`d=2`), on BOTH sides of the
/// crossover (a lean basis where the curved birth pays, and a rich basis where
/// its dictionary surcharge sinks it).
#[test]
fn birth_prescreen_verdict_agrees_with_full_eq4_bits() {
    // Grid of (d, N, P, G): circle and sphere at a 2048-atom overcomplete
    // dictionary, enough rows for stable covariance/SVD spectra.
    for &(d, n, p, g_dict) in &[(1usize, 300usize, 6usize, 2048usize), (2, 320, 8, 2048)] {
        // WIN side: a lean basis (few harmonics) → the support saving from
        // collapsing s active slots to one dwarfs the small dictionary surcharge.
        let lean_m = 2 * d + 1;
        let (eq4_win, pred_win) = eq4_curved_advantage_and_prescreen(d, n, p, g_dict, lean_m);
        assert!(
            eq4_win > 0.0 && pred_win > 0.0,
            "d={d}: lean curved birth must win on BOTH scoreboards, \
             got Eq-4 advantage {eq4_win} and pre-screen {pred_win}"
        );

        // LOSE side: a rich basis (large m) → the (m−s)·P·½log₂N dictionary
        // surcharge overwhelms the support saving on both scoreboards.
        let rich_m = 400;
        let (eq4_lose, pred_lose) = eq4_curved_advantage_and_prescreen(d, n, p, g_dict, rich_m);
        assert!(
            eq4_lose < 0.0 && pred_lose < 0.0,
            "d={d}: rich curved birth must lose on BOTH scoreboards, \
             got Eq-4 advantage {eq4_lose} and pre-screen {pred_lose}"
        );

        // The pre-screen never claims a win the full scorer denies (its raison
        // d'être: it may only DEFER, never spuriously accept).
        assert_eq!(
            eq4_win.signum(),
            pred_win.signum(),
            "d={d}: win-side verdict sign must agree"
        );
        assert_eq!(
            eq4_lose.signum(),
            pred_lose.signum(),
            "d={d}: lose-side verdict sign must agree"
        );
    }
}

/// #2233 end-to-end selection invariance: a RACE of several birth candidates
/// straddling the crossover, adjudicated two ways on the SAME planted fixtures —
///
/// * the **unscreened race**: every candidate is scored through the production
///   Eq-4 fixed-distortion scorer and the winner is `argmax` of the true bits
///   advantage (this is the expensive full refit a birth would trigger);
/// * the **screened race**: the closed-form [`predicted_birth_dl_bits`]
///   pre-screen ranks the same candidates from spectra alone (no refit).
///
/// The pre-screen's whole contract is that it changes the COST of selection (it
/// skips the refit of candidates that cannot win) without changing the OUTCOME.
/// This asserts exactly that: the two races pick the **same winner** and admit
/// the **same set** of candidates. Per-candidate sign agreement
/// ([`birth_prescreen_verdict_agrees_with_full_eq4_bits`]) is necessary but not
/// sufficient for this — two scorers can agree on every sign yet disagree on the
/// `argmax` among the winners, which would silently change which atom is born.
///
/// The candidate merit is graded purely by dictionary overcompleteness `G`: the
/// support saving `(s−1)·log₂(G/L0)` (and the Eq-4 `log₂C(G,s)−log₂G` it prices)
/// is strictly increasing in `G`, so the largest-`G` circle is the unambiguous
/// winner for BOTH scorers, and a basis-rich candidate is the unambiguous loser
/// for both — a race whose ordering is analytically pinned, not incidental.
#[test]
fn birth_prescreen_selects_same_winner_as_unscreened_eq4_race() {
    // (label, d, n, p, g_dict, basis_m). Circle kind (d=1, s=2) at three graded
    // dictionary sizes — the support win rises with G, so candidate 2 (G=8192) is
    // the crossover winner — plus one basis-rich control whose dictionary
    // surcharge sinks it on both scoreboards.
    let race: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("circle_G512", 1, 300, 6, 512, 3),
        ("circle_G2048", 1, 300, 6, 2048, 3),
        ("circle_G8192", 1, 300, 6, 8192, 3),
        ("circle_rich", 1, 300, 6, 2048, 400),
    ];
    let winner_idx = 2; // circle_G8192, the largest-support candidate.

    let mut eq4_adv = Vec::with_capacity(race.len());
    let mut pred = Vec::with_capacity(race.len());
    for &(_label, d, n, p, g_dict, basis_m) in race {
        let (adv, pr) = eq4_curved_advantage_and_prescreen(d, n, p, g_dict, basis_m);
        eq4_adv.push(adv);
        pred.push(pr);
    }

    let argmax = |v: &[f64]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .expect("non-empty race")
    };
    let unscreened_winner = argmax(&eq4_adv);
    let screened_winner = argmax(&pred);

    // Same winner: the pre-screen's argmax is the full Eq-4 race's argmax.
    assert_eq!(
        screened_winner, unscreened_winner,
        "screened pre-screen winner (idx {screened_winner}, {}) must equal the \
         unscreened Eq-4 race winner (idx {unscreened_winner}, {}); \
         eq4_adv={eq4_adv:?} pred={pred:?}",
        race[screened_winner].0, race[unscreened_winner].0
    );
    assert_eq!(
        screened_winner, winner_idx,
        "the largest-G circle must be the analytically-pinned winner; got {}",
        race[screened_winner].0
    );

    // Same admitted set: the candidates the pre-screen proposes (predicted ΔMDL
    // > 0) are exactly the candidates the full Eq-4 race keeps (advantage > 0) —
    // the pre-screen never drops a true winner nor admits a true loser.
    for (i, (&(label, ..), (&adv, &pr))) in race
        .iter()
        .zip(eq4_adv.iter().zip(pred.iter()))
        .enumerate()
    {
        assert_eq!(
            adv > 0.0,
            pr > 0.0,
            "candidate {i} ({label}): screened admit ({}) must match unscreened \
             keep ({}); eq4_adv={adv} pred={pr}",
            pr > 0.0,
            adv > 0.0,
        );
    }
    // And the race is a genuine mix (at least one keep and one drop), so the
    // invariance above is non-trivially exercised.
    assert!(
        eq4_adv.iter().any(|&a| a > 0.0) && eq4_adv.iter().any(|&a| a < 0.0),
        "the race must contain both a winner and a loser; eq4_adv={eq4_adv:?}"
    );
}

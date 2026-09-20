//! Parity tests: the Rust description-length surface must reproduce the
//! hand-verified `Manifold-SAE experiments/mdl_ladder/mdl.py` reference numbers
//! exactly.

use super::{
    BirthMdlPrescreen, BirthPriorityInconclusive, BirthProposalPriority, DecoderBlockCode,
    DescriptionLengthScoreKind, DictionaryCode, DictionaryCodeKind, NativeDescriptionLengthRequest,
    ScoreComparison, ScoredBits, atom_occupancy, birth_proposal_priority, circle_phase_code,
    circle_phase_code_at, description_length_delta, manifold_fit_description_length, matched_dl,
    matched_dl_delta, native_manifold_description_length, persisted_decoder_dictionary_code,
    reverse_water_filling, scalar_rate_bits, se_resolution_bits, selection_bits,
    weighted_reverse_water_filling,
};
use crate::atom_codes::SparseAtomCodes;
use crate::native_code_source::NativeGateModel;
use crate::manifold::{
    SaeAtomBasisKind, SaeAtomGeometryPlan, SaeBasisResolution, SaeReferenceMetricPlan,
};
use ndarray::Array2;

/// The finite priority of a firing candidate with a positive noise floor.
fn priority_bits(p: &BirthMdlPrescreen) -> f64 {
    birth_proposal_priority(p)
        .bits()
        .expect("a firing candidate with a positive noise floor has a finite priority")
}

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

fn declared(n_params: usize, bits_per_scalar: f64) -> DictionaryCode {
    DictionaryCode::DeclaredPrecision {
        n_params,
        bits_per_scalar,
    }
}

fn assert_relative(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: expected {expected}, got {actual}"
    );
}

#[test]
fn manifold_fit_dl_decomposes_and_sums_to_total() {
    // N=40 tokens, G=4 scalar atoms with their own spectra, coded to expected
    // per-token distortion 0.2, 96 decoder scalars at a declared 16 bits.
    let codes = planted_codes(40, 4);
    let spectra = vec![vec![1.0_f64], vec![0.5], vec![0.25], vec![0.125]];
    let delta2 = 0.2;
    let dl = manifold_fit_description_length(&codes, &spectra, delta2, 0.9, &declared(96, 16.0))
        .unwrap();

    // Atom k fires on every (k+1)-th token: occupancies 40/40, 20/40, 14/40, 10/40.
    let occupancy = atom_occupancy(&codes);
    assert_eq!(occupancy, vec![1.0, 0.5, 0.35, 0.25]);
    assert_eq!(dl.atom_occupancy, occupancy);
    let components: Vec<(f64, Vec<f64>)> = occupancy
        .iter()
        .zip(&spectra)
        .map(|(&probability, spectrum)| (probability, spectrum.clone()))
        .collect();
    let rates = weighted_reverse_water_filling(&components, delta2).unwrap();
    for (atom, &rate) in rates.iter().enumerate() {
        assert_relative(dl.atom_code_bits_per_token[atom], rate, 1e-12, "per-atom code bits");
    }
    assert_relative(dl.code_bits_per_token, rates.iter().sum(), 1e-12, "code bits per token");

    // Selection = empirical support entropy H(S).
    let support = codes.support_entropy();
    assert!((dl.selection_bits_per_token - support.tree_bits).abs() < 1e-12);

    // The mean rate per transmitted coordinate reconciles with the code ledger.
    let coded_scalars: f64 = codes
        .iter()
        .map(|c| c.active_mask.count_ones() as f64)
        .sum();
    assert_relative(dl.code_bits, dl.coordinate_rate_bits * coded_scalars, 1e-12, "mean rate");

    // Corpus totals and per-token accounting reconcile with the parts.
    assert!((dl.selection_bits - 40.0 * dl.selection_bits_per_token).abs() < 1e-9);
    assert_eq!(dl.dictionary_code, DictionaryCodeKind::DeclaredPrecision);
    assert!((dl.dict_bits - 96.0 * 16.0).abs() < 1e-12);
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
    // The rate–distortion trade: a finer distortion (smaller budget) costs MORE
    // code bits, from the actual per-atom spectra.
    let codes = planted_codes(30, 3);
    let spectra = vec![vec![1.0_f64], vec![0.6], vec![0.3]];
    let coarse =
        manifold_fit_description_length(&codes, &spectra, 0.3, 0.5, &declared(64, 16.0)).unwrap();
    let fine =
        manifold_fit_description_length(&codes, &spectra, 0.05, 0.95, &declared(64, 16.0)).unwrap();
    assert!(
        fine.coordinate_rate_bits > coarse.coordinate_rate_bits,
        "a tighter distortion must cost more per-coordinate bits: {} !> {}",
        fine.coordinate_rate_bits,
        coarse.coordinate_rate_bits
    );
    assert!(fine.code_bits_per_token > coarse.code_bits_per_token);
}

#[test]
fn manifold_fit_dl_tight_distortion_matches_hand_solved_weighted_rate() {
    // Atom 0 fires on all 10 tokens, atom 1 on the 5 even tokens: occupancies
    // (1, 0.5). Below both variances D(θ) = θ + 0.5θ, so D = 1e-6 gives θ = 1e-6/1.5.
    let codes = planted_codes(10, 2);
    let dl = manifold_fit_description_length(
        &codes,
        &[vec![1.0], vec![0.5]],
        1.0e-6,
        0.999,
        &declared(8, 16.0),
    )
    .unwrap();
    let theta = 1.0e-6_f64 / 1.5;
    let expected = 0.5 * (1.0 / theta).log2() + 0.5 * 0.5 * (0.5 / theta).log2();
    assert_relative(dl.code_bits_per_token, expected, 1e-12, "tight-distortion code bits");
}

#[test]
fn manifold_fit_dl_declared_precision_is_used_verbatim() {
    // A declared fp16 precision (16 bits/scalar) prices the dictionary directly
    // and spends none of the distortion budget.
    let codes = planted_codes(20, 2);
    let dl = manifold_fit_description_length(
        &codes,
        &[vec![1.0], vec![0.5]],
        0.2,
        0.8,
        &declared(50, 16.0),
    )
    .unwrap();
    assert!((dl.l_param_bits - 16.0).abs() < 1e-12);
    assert!((dl.dict_bits - 50.0 * 16.0).abs() < 1e-9);
    assert_eq!(dl.dictionary_header_bits, 0.0);
    assert_eq!(dl.dictionary_distortion, 0.0);
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
    let ceil = (1.0_f64 / 12.0).sqrt();
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
    let columns = 2 * h as i64 + 1; // a circle chart of order H charges 2H+1 = 7 columns

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
    let delta = matched_dl_delta(&flat, &dl).expect("matched reports share one score kind");
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
    let economy_delta =
        matched_dl_delta(&block, &dl).expect("matched reports share one score kind");
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

    // flat − chart, bits
    let corrected_delta =
        matched_dl_delta(&flat, &chart).expect("matched reports share one score kind");

    // The OLD biased arithmetic priced the flat amplitudes at the PHASE SE too.
    let flat_biased = matched_dl(b, b, p, l_param, &phase_ses, ev);
    let biased_delta =
        matched_dl_delta(&flat_biased, &chart).expect("matched reports share one score kind");

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

/// #2933 F23: the circle phase code's exact distortion against ACTUAL quantized
/// reconstructions, over coarse and fine codebooks and a radial spread independent
/// of phase. The planted cloud is a product grid — `n_θ` phases at the centres of
/// equal arcs times two radii `a ± b` — so every tested `M` divides `n_θ` into
/// cells of `m = n_θ/M` points. Each point is encoded to its cell index and decoded
/// to the code's own reconstruction radius at the cell centre, and the measured mean
/// squared error is compared with `D_M = Var(r) + E[r]²·(1 − sinc²(π/M))`.
///
/// The only admissible gap is the grid's in-cell average of `cos(θ − φ_k)`,
/// `S_m = sin(π/M)/(m·sin(π/(Mm)))` instead of `sinc(π/M)`. From
/// `m·sin(x/m) ≥ x − x³/(6m²)` it lies in
/// `[sinc x, sinc x/(1 − x²/(6m²))]`, so the measured error differs from `D_M` by
/// at most `2ρ·E[r]·(S_m − sinc x)`: the test's tolerance, derived, not tuned.
///
/// The old expansion `π²E[r]²/(3M²)` says 0.8225 at M = 2 on a unit circle; the
/// centroid codec measures `1 − (2/π)² = 0.5947`, and the check below would fail
/// the expansion at every coarse codebook.
#[test]
fn circle_phase_code_matches_measured_reconstructions_2933() {
    use std::f64::consts::{PI, TAU};
    let n_theta = 3072 * 8;
    for &(a, b) in &[(1.0_f64, 0.0_f64), (1.5, 0.3)] {
        let radii = [a - b, a + b];
        let mean_radius = a;
        let radial_variance = b * b;
        let mean_square = mean_radius * mean_radius + radial_variance;
        for &m_cells in &[1_u64, 2, 3, 4, 16, 64, 1024] {
            let code = circle_phase_code_at(mean_radius, radial_variance, m_cells);
            let cell = TAU / m_cells as f64;
            let mut squared_error = 0.0_f64;
            let mut count = 0.0_f64;
            for j in 0..n_theta {
                let theta = TAU * (j as f64 + 0.5) / n_theta as f64;
                let k = ((theta / cell).floor() as u64).min(m_cells - 1);
                let phi = (k as f64 + 0.5) * cell;
                for &r in &radii {
                    let dx = r * theta.cos() - code.reconstruction_radius * phi.cos();
                    let dy = r * theta.sin() - code.reconstruction_radius * phi.sin();
                    squared_error += dx * dx + dy * dy;
                    count += 1.0;
                }
            }
            let measured = squared_error / count;
            let x = PI / m_cells as f64;
            let per_cell = (n_theta as u64 / m_cells) as f64;
            let sinc = x.sin() / x;
            let grid_shift = x * x / (6.0 * per_cell * per_cell);
            let tolerance = 2.0 * code.reconstruction_radius.abs() * mean_radius * sinc
                * grid_shift
                / (1.0 - grid_shift)
                + 1.0e-12 * mean_square;
            assert!(
                (measured - code.distortion).abs() <= tolerance,
                "a={a}, b={b}, M={m_cells}: measured {measured} vs exact {} (tolerance {tolerance})",
                code.distortion
            );
            assert_eq!(code.rate_bits, (m_cells as f64).log2());
        }
    }
    // The coarse case the small-cell expansion gets wrong, pinned in closed form.
    let two_cells = circle_phase_code_at(1.0, 0.0, 2);
    let exact = 1.0 - (2.0 / PI) * (2.0 / PI);
    assert!((two_cells.distortion - exact).abs() < 1.0e-12);
    assert!(PI * PI / 12.0 - two_cells.distortion > 0.2);
}

/// #2933 F23: `circle_phase_code` returns the LEAST codebook meeting the budget,
/// makes the zero-rate transition, refuses budgets no phase code can meet, and
/// resolves small-cell distortions at the resolution floor, where the naive
/// `1 − (sin x/x)²` cancels to rounding noise.
#[test]
fn circle_phase_code_selects_the_least_codebook_2933() {
    // Zero rate: a budget at or above E[r²] decodes every firing to the origin.
    let zero = circle_phase_code(1.5, 0.09, 1.5 * 1.5 + 0.09)
        .expect("valid inputs")
        .expect("the origin meets E[r²]");
    assert_eq!(zero.codebook_size, 1);
    assert_eq!(zero.rate_bits, 0.0);
    assert_eq!(zero.reconstruction_radius.abs() < 1.0e-15, true);
    // No finite codebook removes the radial spread.
    assert_eq!(circle_phase_code(1.5, 0.09, 0.09).expect("valid inputs"), None);
    assert_eq!(circle_phase_code(1.5, 0.09, 0.05).expect("valid inputs"), None);
    // A zero mean radius leaves every codebook at E[r²].
    assert_eq!(circle_phase_code(0.0, 0.5, 0.4).expect("valid inputs"), None);
    // Domain errors are loud.
    assert!(circle_phase_code(f64::NAN, 0.0, 0.1).is_err());
    assert!(circle_phase_code(1.0, -0.1, 0.1).is_err());
    assert!(circle_phase_code(1.0, 0.0, f64::INFINITY).is_err());

    // Least codebook: the chosen M meets the budget and M − 1 does not.
    for &budget in &[0.9_f64, 0.6, 0.3, 0.05, 1.0e-3, 1.0e-6] {
        let code = circle_phase_code(1.0, 0.0, budget)
            .expect("valid inputs")
            .expect("a zero-spread ring has a code for every positive budget");
        assert!(code.distortion <= budget, "budget {budget}: {code:?}");
        if code.codebook_size > 1 {
            let coarser = circle_phase_code_at(1.0, 0.0, code.codebook_size - 1);
            assert!(
                coarser.distortion > budget,
                "budget {budget}: M={} is not least ({coarser:?})",
                code.codebook_size
            );
        }
    }

    // Resolution floor. At x = π/M ≈ 1e-8 the exact `1 − sinc²x = x²/3 − 2x⁴/45 + …`
    // is x²/3 to relative 1e-17, while `1 − (sin x/x)²` rounds to 0 or 2.2e-16.
    let budget = 3.0e-16;
    let code = circle_phase_code(1.0, 0.0, budget)
        .expect("valid inputs")
        .expect("a positive budget has a code");
    let x = std::f64::consts::PI / code.codebook_size as f64;
    let small_cell = x * x / 3.0;
    assert!(
        (code.distortion - small_cell).abs() <= 1.0e-12 * small_cell,
        "distortion {} vs x²/3 {small_cell} at M={}",
        code.distortion,
        code.codebook_size
    );
    let coarser = std::f64::consts::PI / (code.codebook_size - 1) as f64;
    assert!(code.distortion <= budget && coarser * coarser / 3.0 > budget);
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
    let (rate, per) = reverse_water_filling(&[1.0, 0.5, 0.1], 0.3).unwrap();
    assert!(close(rate, 2.821928094, 1e-4), "rate {rate}");
    assert!(close(per[0], 1.660964047, 1e-4));
    assert!(close(per[1], 1.160964047, 1e-4));
    assert!(close(per[2], 0.0, 1e-6));
}

#[test]
fn reverse_water_filling_spends_one_total_distortion_budget() {
    let (rate, per) = reverse_water_filling(&[1.0, 1.0], 0.5).unwrap();
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

/// Two tokens-by-atoms supports for the #2933 F13 regressions: `n` rows, atom 0
/// fires on the first `frequent` rows and atom 1 on the first `rare` rows.
fn two_atom_codes(n: usize, frequent: usize, rare: usize) -> SparseAtomCodes {
    let mut codes = SparseAtomCodes::empty(n, 2);
    for row in 0..frequent {
        codes.row_mut(row).assign(0, 1.0);
    }
    for row in 0..rare {
        codes.row_mut(row).assign(1, 1.0);
    }
    codes
}

#[test]
fn native_code_rates_stay_with_the_firing_atom_2933_f13() {
    // Occupancies (1, 0.01). The audit's counterexample: averaging the rates and
    // multiplying by the firing-weighted coordinate count gives the same number
    // whichever atom carries the wide spectrum.
    let codes = two_atom_codes(100, 100, 1);
    let wide_frequent = manifold_fit_description_length(
        &codes,
        &[vec![16.0], vec![1.0e-4]],
        1.0e-2,
        0.9,
        &declared(0, 0.0),
    )
    .unwrap();
    let wide_rare = manifold_fit_description_length(
        &codes,
        &[vec![1.0e-4], vec![16.0]],
        1.0e-2,
        0.9,
        &declared(0, 0.0),
    )
    .unwrap();
    // Hand-solved allocations, D = 0.01 per token.
    // Wide on the frequent atom: D(θ) = θ + 0.01·1e-4, θ = 0.009999.
    let expected_frequent = 0.5 * (16.0_f64 / 0.009999).log2();
    // Wide on the rare atom: D(θ) = 1e-4 + 0.01·θ, θ = 0.99.
    let expected_rare = 0.01 * 0.5 * (16.0_f64 / 0.99).log2();
    assert_relative(wide_frequent.code_bits_per_token, expected_frequent, 1e-9, "wide frequent");
    assert_relative(wide_rare.code_bits_per_token, expected_rare, 1e-9, "wide rare");
    assert_eq!(wide_frequent.atom_code_bits_per_token[1], 0.0);
    assert_eq!(wide_rare.atom_code_bits_per_token[0], 0.0);
    assert!(
        wide_frequent.code_bits_per_token > 100.0 * wide_rare.code_bits_per_token,
        "swapping spectra between a frequent and a rare atom must change the code: {} vs {}",
        wide_frequent.code_bits_per_token,
        wide_rare.code_bits_per_token
    );
}

#[test]
fn native_multidimensional_atoms_match_hand_solved_weighted_allocation_2933_f13() {
    // N = 4: atom 0 (d = 2) fires on 2 tokens, atom 1 (d = 3) on 1 token, so the
    // occupancies are (0.5, 0.25) and the dimensions differ.
    let codes = two_atom_codes(4, 2, 1);
    let two_dim = vec![4.0_f64, 1.0];
    let three_dim = vec![2.0_f64, 0.5, 0.1];
    let budget = 1.0875_f64;

    // θ ∈ [0.5, 1]: D = 0.25·0.1 + 0.25·0.5 + θ·(0.5 + 0.25 + 0.5) = 0.15 + 1.25θ = 1.0875.
    let planted = manifold_fit_description_length(
        &codes,
        &[two_dim.clone(), three_dim.clone()],
        budget,
        0.5,
        &declared(0, 0.0),
    )
    .unwrap();
    let theta = 0.75_f64;
    let atom0 = 0.5 * 0.5 * ((4.0 / theta).log2() + (1.0 / theta).log2());
    let atom1 = 0.25 * 0.5 * (2.0 / theta).log2();
    assert_relative(planted.atom_code_bits_per_token[0], atom0, 1e-12, "atom 0");
    assert_relative(planted.atom_code_bits_per_token[1], atom1, 1e-12, "atom 1");
    assert_relative(planted.coord_dim, (0.5 * 2.0 + 0.25 * 3.0) / 0.75, 1e-12, "mean dim");

    // Swapped: θ ∈ [0.5, 1]: D = 0.5·(θ + 0.5 + 0.1) + 0.25·(θ + θ) = θ + 0.3 = 1.0875.
    let swapped = manifold_fit_description_length(
        &codes,
        &[three_dim, two_dim],
        budget,
        0.5,
        &declared(0, 0.0),
    )
    .unwrap();
    let theta = 0.7875_f64;
    let atom0 = 0.5 * 0.5 * (2.0 / theta).log2();
    let atom1 = 0.25 * 0.5 * ((4.0 / theta).log2() + (1.0 / theta).log2());
    assert_relative(swapped.atom_code_bits_per_token[0], atom0, 1e-12, "swapped atom 0");
    assert_relative(swapped.atom_code_bits_per_token[1], atom1, 1e-12, "swapped atom 1");
    assert!(
        (planted.code_bits_per_token - swapped.code_bits_per_token).abs()
            > 0.1 * planted.code_bits_per_token,
        "swapping multidimensional spectra must change the code: {} vs {}",
        planted.code_bits_per_token,
        swapped.code_bits_per_token
    );
}

#[test]
fn quantized_decoder_shares_the_budget_by_hand_2933_f14() {
    // N = 4 tokens, one scalar atom firing everywhere with variance 1. One decoder
    // block: range 2, one output channel, row sensitivity 0.75, so a coefficient's
    // output variance is v = 0.75·2²/12 = 0.25 and it enters as weight 1/4 on N·v = 1.
    // D(θ) = min(1, θ) + ¼·min(1, θ) = 1.25θ, so D = 0.5 gives θ = 0.4.
    let codes = two_atom_codes(4, 4, 0);
    let dictionary = DictionaryCode::Quantized {
        blocks: vec![DecoderBlockCode {
            coefficient_range: 2.0,
            p_out: 1,
            row_sensitivity: vec![0.75],
        }],
        header_bits: 128.0,
    };
    let dl = manifold_fit_description_length(&codes, &[vec![1.0], vec![]], 0.5, 0.5, &dictionary)
        .unwrap();
    let theta = 0.4_f64;
    assert_relative(dl.code_bits_per_token, 0.5 * (1.0 / theta).log2(), 1e-12, "code bits");
    // The coefficient spends θ/N = 0.1 of output distortion: ½log₂(v/0.1) bits.
    let coefficient_bits = 0.5 * (0.25_f64 / 0.1).log2();
    assert_eq!(dl.dictionary_code, DictionaryCodeKind::Quantized);
    assert_eq!(dl.n_params, 1);
    assert_relative(dl.dictionary_distortion, 0.1, 1e-12, "dictionary distortion");
    assert_relative(dl.l_param_bits, coefficient_bits, 1e-12, "coefficient bits");
    assert_relative(dl.dict_bits, coefficient_bits + 128.0, 1e-12, "dictionary bits");
    assert_eq!(dl.dictionary_header_bits, 128.0);
}

#[test]
fn quantized_decoder_is_never_free_for_constant_coordinates_2933_f14() {
    // Fixed latent codes: one periodic atom at a constant coordinate (zero code
    // variance) firing on every token. Only the decoder carries information.
    let plan = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Periodic,
        1,
        SaeBasisResolution::PeriodicHarmonics { order: 1 },
        SaeReferenceMetricPlan::UnitCircle,
    )
    .unwrap();
    let basis_size = plan.basis_size().unwrap();
    let n = 32;
    let coords = Array2::<f64>::from_elem((n, 1), 0.125);
    let assignments = Array2::<f64>::ones((n, 1));
    let mut codes = SparseAtomCodes::empty(n, 1);
    for row in 0..n {
        codes.row_mut(row).assign(0, 1.0);
    }
    let decoder =
        Array2::from_shape_fn((basis_size, 2), |(m, c)| 0.25 * (m as f64 + 1.0) - 0.5 * c as f64);
    let mut moved = decoder.clone();
    moved[[0, 1]] += 3.0;
    let price = |block: &Array2<f64>| {
        let dictionary = persisted_decoder_dictionary_code(
            std::slice::from_ref(&plan),
            &[block.view()],
            &[coords.view()],
            assignments.view(),
            None,
            2,
        )
        .unwrap();
        manifold_fit_description_length(&codes, &[vec![0.0]], 1.0e-3, 0.99, &dictionary).unwrap()
    };
    let base = price(&decoder);
    let edited = price(&moved);
    assert_eq!(base.code_bits, 0.0, "a constant coordinate carries no code bits");
    assert_eq!(base.n_params as usize, basis_size * 2);
    let base_coefficients = base.dict_bits - base.dictionary_header_bits;
    let edited_coefficients = edited.dict_bits - edited.dictionary_header_bits;
    assert!(
        base_coefficients > 1.0,
        "{} decoder coefficients were priced at {base_coefficients} bits",
        basis_size * 2
    );
    assert!(
        edited_coefficients > base_coefficients + 1.0,
        "widening a decoder coefficient must cost more bits: {edited_coefficients} vs {base_coefficients}"
    );
    assert!(base.dictionary_header_bits > 0.0);
    // The codes are constant, so the decoder spends the whole budget.
    assert_relative(base.dictionary_distortion, 1.0e-3, 1e-9, "dictionary distortion");
}

#[test]
fn quantized_decoder_is_priced_in_the_standardized_output_metric_2933_f14() {
    // The fit's EV and the native code spectra live in the metric diag(σ⁻²).
    // Rescaling one physical decoder column together with its tier0 scale leaves
    // the standardized model unchanged, so the decoder message must not move.
    let plan = SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Periodic,
        1,
        SaeBasisResolution::PeriodicHarmonics { order: 1 },
        SaeReferenceMetricPlan::UnitCircle,
    )
    .unwrap();
    let basis_size = plan.basis_size().unwrap();
    let n = 16;
    let coords = Array2::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let assignments = Array2::<f64>::ones((n, 1));
    let mut codes = SparseAtomCodes::empty(n, 1);
    for row in 0..n {
        codes.row_mut(row).assign(0, 1.0);
    }
    let decoder =
        Array2::from_shape_fn((basis_size, 2), |(m, c)| 0.3 * (m as f64 + 1.0) - 0.4 * c as f64);
    let scale = ndarray::array![1.0, 0.5];
    let mut physical = decoder.clone();
    physical.column_mut(1).mapv_inplace(|value| 40.0 * value);
    let compensated = ndarray::array![1.0, 20.0];
    let price = |block: &Array2<f64>, tier0: &ndarray::Array1<f64>| {
        let dictionary = persisted_decoder_dictionary_code(
            std::slice::from_ref(&plan),
            &[block.view()],
            &[coords.view()],
            assignments.view(),
            Some(tier0.view()),
            2,
        )
        .unwrap();
        manifold_fit_description_length(&codes, &[vec![0.0]], 1.0e-3, 0.99, &dictionary).unwrap()
    };
    let base = price(&decoder, &scale);
    let moved = price(&physical, &compensated);
    assert_relative(moved.dict_bits, base.dict_bits, 1e-9, "compensated decoder bits");
    assert_relative(
        moved.dictionary_distortion,
        base.dictionary_distortion,
        1e-9,
        "compensated decoder distortion",
    );
    // Control: the same physical rescaling without its scale is a different model.
    let uncompensated = price(&physical, &scale);
    assert!(
        (uncompensated.dict_bits - base.dict_bits).abs() > 1.0,
        "an uncompensated rescaling must change the decoder message: {} vs {}",
        uncompensated.dict_bits,
        base.dict_bits
    );
}

#[test]
fn water_filling_rejects_nonfinite_and_non_psd_spectra_2933_f44() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(reverse_water_filling(&[bad, 1.0], 0.5).is_err(), "eigenvalue {bad}");
        assert!(
            weighted_reverse_water_filling(&[(1.0, vec![1.0, bad])], 0.5).is_err(),
            "weighted eigenvalue {bad}"
        );
        assert!(
            weighted_reverse_water_filling(&[(bad, vec![1.0])], 0.5).is_err(),
            "weight {bad}"
        );
        assert!(reverse_water_filling(&[1.0], bad).is_err(), "budget {bad}");
    }
    assert!(reverse_water_filling(&[1.0], -1.0e-12).is_err());
    assert!(weighted_reverse_water_filling(&[(-0.5, vec![1.0])], 0.5).is_err());

    // A rounded zero eigenvalue is clipped; a material negative one is reported.
    let (rate, per) = reverse_water_filling(&[1.0, -1.0e-17], 0.25).unwrap();
    assert_relative(rate, 0.5 * (1.0_f64 / 0.25).log2(), 1e-12, "rounding-clipped rate");
    assert_eq!(per[1], 0.0);
    let error = reverse_water_filling(&[1.0, -1.0e-3], 0.25).unwrap_err();
    assert!(error.contains("not positive semidefinite"), "{error}");
    assert!(reverse_water_filling(&[-1.0e-20], 0.25).is_err());
}

#[test]
fn zero_distortion_is_an_infinite_rate_not_an_error_2933_f44() {
    let (rate, per) = reverse_water_filling(&[1.0, 0.0], 0.0).unwrap();
    assert!(rate.is_infinite() && rate > 0.0);
    assert_eq!(per[1], 0.0);
    let rates =
        weighted_reverse_water_filling(&[(1.0, vec![2.0]), (0.0, vec![5.0]), (1.0, vec![0.0])], 0.0)
            .unwrap();
    assert!(rates[0].is_infinite());
    assert_eq!(rates[1], 0.0, "a never-firing component transmits nothing");
    assert_eq!(rates[2], 0.0, "a zero-variance component transmits nothing");
    let (empty_rate, empty_per) = reverse_water_filling(&[], 0.0).unwrap();
    assert_eq!(empty_rate, 0.0);
    assert!(empty_per.is_empty());

    // ev = 1 with zero budget: continuous codes need unbounded rate.
    let codes = planted_codes(8, 2);
    let dl =
        manifold_fit_description_length(&codes, &[vec![1.0], vec![0.0]], 0.0, 1.0, &declared(4, 8.0))
            .unwrap();
    assert!(dl.code_bits.is_infinite() && dl.bits_per_token.is_infinite());
}

#[test]
fn fit_description_length_validates_its_domain_2933_f44() {
    let codes = planted_codes(8, 2);
    let spectra = vec![vec![1.0], vec![0.5]];
    let ok = |ev: f64| manifold_fit_description_length(&codes, &spectra, 0.1, ev, &declared(4, 8.0));

    // EV just above one is impossible; just below one, negative, and one are valid.
    assert!(ok(1.0 + 1.0e-12).is_err());
    assert!(ok(f64::NAN).is_err());
    assert!(ok(f64::INFINITY).is_err());
    assert!(ok(f64::NEG_INFINITY).is_err());
    assert!(ok(1.0 - 1.0e-12).is_ok());
    assert!(ok(1.0).is_ok());
    assert!(ok(-2.5).is_ok(), "negative held-out EV is meaningful");

    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(
            manifold_fit_description_length(&codes, &spectra, bad, 0.5, &declared(4, 8.0)).is_err(),
            "budget {bad}"
        );
        assert!(
            manifold_fit_description_length(&codes, &spectra, 0.1, 0.5, &declared(4, bad)).is_err(),
            "declared precision {bad}"
        );
        let block = |range: f64, sensitivity: f64, header: f64| DictionaryCode::Quantized {
            blocks: vec![DecoderBlockCode {
                coefficient_range: range,
                p_out: 2,
                row_sensitivity: vec![sensitivity, 1.0],
            }],
            header_bits: header,
        };
        for dictionary in [block(bad, 1.0, 64.0), block(1.0, bad, 64.0), block(1.0, 1.0, bad)] {
            assert!(
                manifold_fit_description_length(&codes, &spectra, 0.1, 0.5, &dictionary).is_err(),
                "quantized dictionary input {bad}"
            );
        }
    }
    for bad in [f64::NAN, f64::INFINITY, -0.25] {
        assert!(
            manifold_fit_description_length(&codes, &[vec![1.0], vec![bad]], 0.1, 0.5, &declared(4, 8.0))
                .is_err(),
            "spectrum value {bad}"
        );
    }

    // Structural failures: no tokens, a spectrum count that disagrees with G.
    let empty = SparseAtomCodes::empty(0, 2);
    assert!(manifold_fit_description_length(&empty, &spectra, 0.1, 0.5, &declared(4, 8.0)).is_err());
    assert!(manifold_fit_description_length(&codes, &spectra[..1], 0.1, 0.5, &declared(4, 8.0)).is_err());

    // A zero-dimensional atom is valid and transmits no coordinates.
    let point = manifold_fit_description_length(&codes, &[vec![1.0], vec![]], 0.1, 0.5, &declared(4, 8.0))
        .unwrap();
    assert_eq!(point.atom_code_bits_per_token[1], 0.0);
    assert!(point.code_bits_per_token > 0.0);
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
    let got_circle = priority_bits(&circle);
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
    let got_torus = priority_bits(&torus);
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

/// #2233 signed-dictionary regression: a HIGH-CODIMENSION birth whose curved basis
/// is NARROWER than the flat span it replaces (`m < ŝ`) must be CREDITED the exact
/// dictionary saving `+(ŝ−m)·P·½log₂N`, not clamped to zero. This is the theorem's
/// principal win for a manifold spanning many ambient directions on a compact basis
/// (a shell/high-genus kind), which an earlier `(m−ŝ).max(0)` clamp defers
/// indefinitely. The priority only orders proposals and certifies nothing
/// (#2933 F22); the e-process gate stays sole arbiter.
#[test]
fn birth_prescreen_credits_dictionary_saving_when_basis_narrower_than_span() {
    // ŝ=8 ambient span, compact basis m=5 (< ŝ), d=2. N=1000, P=8, G=1024, L0=32.
    let p = BirthMdlPrescreen {
        rho: 0.05,
        span: 8.0,
        intrinsic_dim: 2,
        basis_size: 5,
        signal_var: 3.0,
        noise_floor: 1.0,
        n_tokens: 1000.0,
        p_out: 8,
        g_dict: 1024,
        l0: 32.0,
    };
    let log2_n = 1000.0_f64.log2();
    // saving = ρN·[(ŝ−d−1)·½log₂3 + (ŝ−1)·log₂(G/L0)]
    let code = (8.0 - 2.0 - 1.0) * scalar_rate_bits(3.0, 1.0);
    let support = (8.0 - 1.0) * (1024.0_f64 / 32.0).log2();
    let saving = 0.05 * 1000.0 * (code + support);
    // signed dictionary delta = (m−ŝ)·P·½log₂N = (5−8)·8·½·log₂N  (NEGATIVE ⇒ saving)
    let dict_delta = (5.0 - 8.0) * 8.0 * 0.5 * log2_n;
    let expected = saving - dict_delta; // saving − (negative) = saving + |dict_delta|
    let got = priority_bits(&p);
    assert!(
        (got - expected).abs() < 1e-9,
        "signed-dictionary bits {got} != hand value {expected}"
    );
    // The dictionary term is a genuine SAVING: the prediction strictly exceeds the
    // code+support saving alone (the old clamp would have returned exactly `saving`).
    assert!(
        got > saving + 1.0,
        "a narrower-than-span basis (m<ŝ) must be CREDITED the dictionary saving: \
         got {got}, saving-alone {saving} (a zero-clamp regression)"
    );
    // Monotonicity: an even narrower basis earns a strictly larger saving.
    let narrower = priority_bits(&BirthMdlPrescreen { basis_size: 3, ..p });
    assert!(
        narrower > got,
        "a narrower basis must earn a larger dictionary saving: m=3 {narrower} <= m=5 {got}"
    );
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
/// path — and the heuristic birth priority on the SAME planted quantities. On these
/// integer-span fixtures both carry the same sign; in general they need not
/// ([`birth_priority_is_not_an_eq4_certificate_2933`]).
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
            // Both featurizers declare the same planted N-token amortisation
            // horizon (matching the pre-screen's `n_tokens = n`), so the
            // flat-vs-curved advantage isolates the support/dictionary crossover.
            n as i64,
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
            n as i64,
            &[target],
            None,
            move |_, take| {
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

    let predicted = priority_bits(&BirthMdlPrescreen {
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

/// #2233 agreement on integer-span fixtures: the birth priority's sign matches the
/// full Eq-4 fixed-distortion bits computation run through the production scorer
/// on the same planted `s=d+1` data — for a circle (`d=1`) and a sphere (`d=2`), on
/// BOTH sides of the crossover (a lean basis where the curved birth pays, and a
/// rich basis where its dictionary surcharge sinks it). This is agreement on these
/// fixtures, not a guarantee (#2933 F22).
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

        // On these fixtures the two scoreboards agree in sign.
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
/// * the **screened race**: the heuristic [`birth_proposal_priority`] ranks the
///   same candidates from spectra alone (no refit).
///
/// On these integer-span fixtures the ranking reproduces the outcome: the two races
/// pick the **same winner** and split the candidates into the **same signs**. The
/// priority is not a certificate in general (#2933 F22), so it orders proposals
/// and never excludes them. Per-candidate sign agreement
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
            .max_by(|&(_, a), &(_, b)| a.total_cmp(b))
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

    // Same signs: on these fixtures the candidates with a positive priority are
    // exactly the candidates the full Eq-4 race keeps (advantage > 0).
    for (i, (&(label, ..), (&adv, &pr))) in
        race.iter().zip(eq4_adv.iter().zip(pred.iter())).enumerate()
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

/// #2233 Task 2 — the per-kind crossover table over the six zoo geometry
/// classes. The theorem's split: a kind with ambient span exactly `d+1`
/// (circle, sphere, Möbius, swiss sheet) wins Eq-4 bits through SUPPORT (+
/// residual) ONLY — its code coefficient `s−d−1` is zero — while a kind with
/// `s > d+1` (torus `s=4,d=2`, helix `s=3,d=1`) carries a strictly positive
/// CODE saving on top.
///
/// The code term is the only λ̂-dependent piece of the closed form, so the
/// class split has a sharp empirical discriminator that needs no term
/// plumbing: the predicted bits of a support-only kind are INVARIANT to the
/// birth direction's signal variance, while torus/helix strictly increase in
/// it by exactly `(s−d−1)·ρN·Δ(½log₂(λ̂/δ))`.
#[test]
fn per_kind_crossover_table_splits_code_vs_support_classes_2233() {
    // Shared budget: N=2000 tokens, P=8 channels, G=1024 atoms, L0=32 active,
    // firing rate ρ=0.05, noise floor δ=1.
    let (n_tokens, p_out, g_dict, l0, rho, noise) =
        (2000.0_f64, 8usize, 1024usize, 32.0_f64, 0.05_f64, 1.0_f64);
    let log2_n = n_tokens.log2();
    let log2_g_over_l0 = (g_dict as f64 / l0).log2();

    // (label, span s, intrinsic d, basis m): circle/sphere/torus READ the
    // production span→topology map rather than transcribing it. #2749 — the
    // transcription that used to live here outlived the `(lat, lon)` chart it
    // copied, and went on asserting the class structure at a sphere width (7) no
    // constructor could build. Möbius and the swiss sheet share the sphere
    // band's budget at their common (s=3, d=2); the helix is a d=1 closed curve
    // winding through 3 ambient directions on the circle basis.
    let priced = |span: f64| -> (usize, usize) {
        let plan = crate::manifold::SaeAtomGeometryPlan::curved_prescreen_atom_for_span(span)
            .expect("the production pre-screen map must build its atom");
        (
            plan.intrinsic_dim(),
            plan.basis_size().expect("a built plan has a width"),
        )
    };
    let (circle_d, circle_m) = priced(2.0);
    let (sphere_d, sphere_m) = priced(3.0);
    let (torus_d, torus_m) = priced(4.0);
    let kinds: [(&str, f64, usize, usize); 6] = [
        ("circle", 2.0, circle_d, circle_m),
        ("sphere", 3.0, sphere_d, sphere_m),
        ("mobius", 3.0, sphere_d, sphere_m),
        ("swiss_sheet", 3.0, sphere_d, sphere_m),
        ("torus", 4.0, torus_d, torus_m),
        ("helix", 3.0, 1, circle_m),
    ];

    let prescreen = |span: f64, d: usize, m: usize, signal_var: f64| BirthMdlPrescreen {
        rho,
        span,
        intrinsic_dim: d,
        basis_size: m,
        signal_var,
        noise_floor: noise,
        n_tokens,
        p_out,
        g_dict,
        l0,
    };

    for &(label, span, d, m) in &kinds {
        let code_coefficient = span - d as f64 - 1.0;
        let low = priority_bits(&prescreen(span, d, m, 3.0));
        let high = priority_bits(&prescreen(span, d, m, 300.0));
        let support_only_value = rho * n_tokens * (span - 1.0) * log2_g_over_l0
            - (m as f64 - span) * p_out as f64 * 0.5 * log2_n;
        match label {
            // s = d+1: SUPPORT-only class. No code saving at any signal level,
            // and the closed form reduces exactly to support − dictionary.
            "circle" | "sphere" | "mobius" | "swiss_sheet" => {
                assert!(
                    code_coefficient.abs() < 1e-12,
                    "{label}: span {span} with d={d} must sit exactly at s=d+1"
                );
                assert!(
                    (high - low).abs() < 1e-9,
                    "{label} is support-only; predicted bits must be invariant \
                     to signal variance (low={low}, high={high})"
                );
                assert!(
                    (low - support_only_value).abs() < 1e-9,
                    "{label}: support-only closed form mismatch: got {low}, \
                     expected {support_only_value}"
                );
            }
            // s > d+1: torus and helix carry a strictly positive CODE term.
            "torus" | "helix" => {
                assert!(
                    code_coefficient >= 1.0 - 1e-12,
                    "{label}: span {span} with d={d} must have s−d−1 ≥ 1"
                );
                let rate_low = scalar_rate_bits(3.0, noise);
                let rate_high = scalar_rate_bits(300.0, noise);
                let expected_gain = code_coefficient * rho * n_tokens * (rate_high - rate_low);
                assert!(
                    high - low > 0.0,
                    "{label} must strictly gain code bits with signal variance \
                     (low={low}, high={high})"
                );
                assert!(
                    ((high - low) - expected_gain).abs() < 1e-9,
                    "{label}: code gain {} != (s−d−1)·ρN·Δrate {}",
                    high - low,
                    expected_gain
                );
                assert!(
                    low - support_only_value > 0.0,
                    "{label} must beat its own support-only baseline even at \
                     modest signal (code term strictly positive)"
                );
            }
            _ => unreachable!(),
        }
    }
}

/// The faithful whole-dictionary crossover: for a hybrid dictionary that REPLACES
/// flat atoms with curved ones so the total decoder-parameter count is matched to
/// the flat bar (`K_flat + K_curved·b = K_ext`, the issue's inequality ★ at
/// equality), the Eq-4 bits advantage of the curved single-atom featurizer over
/// the `s`-latent flat alternative on one planted `s=d+1` factor — scored through
/// the production Eq-4 scorer with the SAME declared `dictionary_params` on both
/// arms.
///
/// The per-factor test [`eq4_curved_advantage_and_prescreen`] charges the curved
/// atom its full `basis_m·P` decoder against removing only the `s` flat columns it
/// covers, i.e. a stand-alone `(basis_m−s)·P` SURCHARGE — the "stacked on top"
/// config that cannot win the ~95%-dictionary Eq-4 scoreboard for a rich basis.
/// The FAITHFUL config instead funds the curved atom's `b·P` columns by dropping
/// `b` flat filler atoms elsewhere in the `K_ext` budget, so `dict_params(hybrid)
/// = dict_params(flat)` exactly. The decoder-storage term is then bitwise
/// identical on both arms and CANCELS: the contest is decided by support + code +
/// residual alone. Passing the matched `dictionary_params` (equality is what the
/// whole-dictionary tie guarantees; the magnitude is immaterial since it cancels)
/// isolates that pure support win — the corollary the 32K creditscope run must
/// confirm, which EV-at-matched-actives under-credits.
fn eq4_matched_dictionary_advantage(d: usize, n: usize, p: usize, g_dict: usize) -> f64 {
    use crate::eq4_description_length::eq4_fixed_distortion_description_length;
    use ndarray::Array2;

    let s = d + 1;
    let radius = 2.0_f64;
    let target = 0.9_f64;
    let (noisy, signal) = planted_sd1_signal(n, p, d, radius);
    let test_x = Array2::from_shape_fn((n, p), |(i, j)| noisy[i][j]);
    let recon = Array2::from_shape_fn((n, p), |(i, j)| signal[i][j]);
    let signal_mat = Array2::from_shape_fn((n, p), |(i, j)| signal[i][j]);

    // The matched decoder-storage charge shared by BOTH arms. In the faithful
    // whole-dictionary config the curved atom's `basis_m·P` columns are funded by
    // removing `basis_m` flat filler atoms, so the total decoder count ties the
    // flat bar; the per-factor scorer therefore sees the SAME `dictionary_params`
    // on both arms and the term cancels. Its magnitude is immaterial (any equal
    // value cancels); the flat bar's `s·P` per-factor share is used.
    let matched_dictionary_params = (s * p) as i64;

    // FLAT: s atoms, each firing every row, one coded coordinate.
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
            matched_dictionary_params,
            n as i64,
            &[target],
            None,
            move |atom, take| {
                let mut out = Array2::zeros((take.len(), p));
                for (out_row, &src) in take.iter().enumerate() {
                    out[[out_row, atom]] = signal_mat[[src, atom]];
                }
                Ok(out)
            },
        )
        .expect("flat Eq-4 scoring succeeds")
    };

    // CURVED: one atom firing every row with d+1 coded coordinates. The basis
    // width `basis_m` sets the flat atoms it displaces (`K_flat = K_ext − b`), NOT
    // this arm's declared decoder charge, which is matched to the flat bar.
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
            matched_dictionary_params,
            n as i64,
            &[target],
            None,
            move |_, take| {
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

    // Dictionary bits tie exactly (matched params), so the advantage is pure
    // support + code + residual.
    assert_eq!(
        flat.dictionary_bits, curved.dictionary_bits,
        "matched-config dictionary bits must tie exactly (flat {} vs curved {})",
        flat.dictionary_bits, curved.dictionary_bits
    );
    flat.per_target[0].bits - curved.per_target[0].bits
}

/// #2233 corollary — the faithful matched-dictionary crossover the 32K creditscope
/// run must confirm. At the whole-dictionary config where the hybrid REPLACES flat
/// atoms (so total decoder params tie the flat bar, ★ at equality), the curved
/// atom wins on SUPPORT alone with no dictionary penalty — a strictly larger and
/// robustly positive margin than the per-factor "stacked" config, and positive
/// EVEN for a basis so rich that the stand-alone surcharge sinks it in
/// [`birth_prescreen_verdict_agrees_with_full_eq4_bits`]. Also pins the exact
/// real-scale config (`K_ext=32768, P=2048, top_k=32`) for the GPU measurement:
/// the ★ inequality is an equality and the support term is large and positive.
#[test]
fn faithful_matched_dictionary_hybrid_wins_on_support_alone_2233() {
    for &(d, n, p, g_dict) in &[(1usize, 300usize, 6usize, 2048usize), (2, 320, 8, 2048)] {
        // The matched-config advantage is basis-INDEPENDENT: the decoder-storage
        // term cancels between the arms, so only support + code + residual remain.
        let matched = eq4_matched_dictionary_advantage(d, n, p, g_dict);

        // Lean basis: the stacked per-factor config already wins, but by LESS — the
        // matched config beats it by exactly the refunded dictionary surcharge.
        let lean_m = 2 * d + 1;
        let (surcharged_lean, _) = eq4_curved_advantage_and_prescreen(d, n, p, g_dict, lean_m);
        assert!(
            surcharged_lean > 0.0 && matched > surcharged_lean,
            "d={d}: matched-config advantage {matched} must beat the lean stacked \
             advantage {surcharged_lean} (the refunded dictionary surcharge)"
        );

        // Rich basis: the stacked per-factor config LOSES (its surcharge sinks it),
        // but the faithful matched config still WINS on support alone — the
        // corollary's whole point, and the margin EV-at-matched-actives misses.
        let rich_m = 400;
        let (surcharged_rich, _) = eq4_curved_advantage_and_prescreen(d, n, p, g_dict, rich_m);
        assert!(
            surcharged_rich < 0.0,
            "d={d}: the stacked rich-basis config must lose (surcharge), got {surcharged_rich}"
        );
        assert!(
            matched > 0.0,
            "d={d}: the faithful matched-config hybrid must WIN on support even where \
             a rich stacked basis loses, got {matched}"
        );
    }

    // Real-scale config pin. The load-bearing scalars are SOURCED from #2283's
    // authoritative measurement driver `experiments/1026_close/driver_1026_arms.py`
    // (argparse defaults) so the scoreboard and the GPU measurement certify the
    // SAME config — a drift here would make the corollary vacuous:
    //   --K 32768   --top-k 32   --curved-atoms 256   (P=2048 is the creditscope
    //   L30 residual_post width; the horizon is the run's declared training N, 96k
    //   train of the 120k --max-rows × 0.8; b = 2H+1 is fit-determined for the
    //   circle curved atom, so it is swept, not pinned).
    // The driver's `--k-flat` is a FREE parameter self-certified faithful by
    //   dict_params(hybrid) = k_flat·P + curved_atoms·b·P ≤ K·P   AND   k_flat ≥ top_k
    // (`bits_dict_params_faithful` in the driver). This test uses the TIGHTEST such
    // config, k_flat = K_ext − K_curved·b, at which ★ is an EQUALITY so the
    // decoder-parameter counts — and thus the dictionary bits — tie exactly.
    let (k_ext, p_out) = (32_768_i64, 2_048_i64); // driver --K, creditscope L30 width
    let top_k = 32.0_f64; // driver --top-k
    let driver_curved_atoms = 256_i64; // driver --curved-atoms default
    // The driver default (256) is the centre of the swept range; the corollary is
    // asserted across a band of curved-atom counts and fit-determined harmonics.
    assert!(
        [64_i64, 256, 1024].contains(&driver_curved_atoms),
        "the swept K_curved band must include the driver's pinned --curved-atoms"
    );
    for &k_curved in &[64_i64, 256, 1024] {
        for &harmonics in &[1_i64, 4, 11] {
            let b = 2 * harmonics + 1; // circle Fourier basis width, b = 2H+1
            let k_flat = k_ext - k_curved * b;
            // The driver's two faithfulness conditions must both hold at this config.
            assert!(
                k_flat >= top_k as i64,
                "driver requires k_flat ≥ top_k (K_curved={k_curved}, b={b}): k_flat={k_flat}"
            );
            assert!(
                k_flat * p_out + k_curved * b * p_out <= k_ext * p_out,
                "driver faithfulness ≤ must hold (K_curved={k_curved}, H={harmonics})"
            );
            // At the tightest config ★ holds with EQUALITY: the two decoder-parameter
            // counts are identical, so dictionary bits tie exactly.
            assert_eq!(
                k_flat * p_out + k_curved * b * p_out,
                k_ext * p_out,
                "tightest faithful ★ must be an equality (K_curved={k_curved}, H={harmonics})"
            );
            // Dictionary bits tie exactly at the equality config (any horizon N).
            let horizon = 96_000.0_f64; // creditscope train N
            let dict = |params: i64| 0.5 * params as f64 / horizon * horizon.log2();
            assert_eq!(
                dict(k_flat * p_out + k_curved * b * p_out),
                dict(k_ext * p_out),
                "matched-config dictionary bits must tie at real scale"
            );
            // The support term the win rides on: each circle (s=2) frees s−1=1 of
            // the top_k=32 active slots per active token, worth log₂(G/L0) bits —
            // large and positive at 32K overcompleteness.
            let support_per_freed_slot = (k_ext as f64 / top_k).log2();
            assert!(
                support_per_freed_slot > 9.0,
                "32K support credit per freed slot must be large, got \
                 {support_per_freed_slot}"
            );
            let predicted = priority_bits(&BirthMdlPrescreen {
                rho: 0.1,
                span: 2.0,
                intrinsic_dim: 1,
                basis_size: b as usize,
                signal_var: 2.0,
                noise_floor: 0.05,
                n_tokens: horizon,
                p_out: p_out as usize,
                g_dict: k_ext as usize,
                l0: top_k,
            });
            // The stand-alone pre-screen carries the −(b−s)·P·½log₂N surcharge; the
            // faithful config refunds it, so the matched-config saving is even
            // larger. Confirm the support-only saving (surcharge added back) is
            // strictly positive — the corollary's "wide margin".
            let refunded_surcharge = (b as f64 - 2.0) * p_out as f64 * 0.5 * horizon.log2();
            let matched_saving = predicted + refunded_surcharge;
            assert!(
                matched_saving > 0.0,
                "faithful matched saving must be a strict win at real scale \
                 (K_curved={k_curved}, H={harmonics}): {matched_saving}"
            );
        }
    }
}

/// #2933 F22: where the old closed form returned NaN the priority returns a typed
/// `Inconclusive`, and zero occupancy is simplified before any rate is evaluated.
#[test]
fn birth_priority_is_inconclusive_instead_of_nan_2933() {
    let audit_case = BirthMdlPrescreen {
        rho: 0.1,
        span: 2.0,
        intrinsic_dim: 1,
        basis_size: 3,
        signal_var: 3.0,
        noise_floor: 0.0,
        n_tokens: 1000.0,
        p_out: 8,
        g_dict: 1024,
        l0: 32.0,
    };
    let unbounded = BirthProposalPriority::Inconclusive(BirthPriorityInconclusive::UnboundedCodeRate);
    // ŝ−d−1 = 0 against an unbounded rate was 0·∞ = NaN. Both arms send the same
    // number of scalars at an unbounded rate, so no finite difference exists.
    assert_eq!(birth_proposal_priority(&audit_case), unbounded);
    // A non-integer span on the same zero floor is inconclusive for the same reason.
    assert_eq!(
        birth_proposal_priority(&BirthMdlPrescreen {
            span: 1.5,
            ..audit_case
        }),
        unbounded
    );
    // Zero occupancy transmits nothing: the divergent rate is never reached and the
    // priority is exactly the signed dictionary term.
    let silent = BirthMdlPrescreen {
        rho: 0.0,
        ..audit_case
    };
    let dictionary_only = 0.0 - (3.0 - 2.0) * 8.0 * 0.5 * 1000.0_f64.log2();
    assert_eq!(
        birth_proposal_priority(&silent),
        BirthProposalPriority::Bits(dictionary_only)
    );
    // No signal on a zero floor has a zero rate, not an unbounded one.
    let no_signal = birth_proposal_priority(&BirthMdlPrescreen {
        signal_var: 0.0,
        ..audit_case
    });
    assert!(no_signal.bits().is_some_and(f64::is_finite), "{no_signal:?}");
    // Out-of-domain fields are refused, never clamped into a number.
    let invalid = BirthProposalPriority::Inconclusive(BirthPriorityInconclusive::InvalidInput);
    for bad in [
        BirthMdlPrescreen {
            rho: f64::NAN,
            ..audit_case
        },
        BirthMdlPrescreen {
            rho: 1.5,
            ..audit_case
        },
        BirthMdlPrescreen {
            span: f64::INFINITY,
            ..audit_case
        },
        BirthMdlPrescreen {
            noise_floor: -1.0,
            ..audit_case
        },
        BirthMdlPrescreen {
            signal_var: f64::NAN,
            ..audit_case
        },
    ] {
        assert_eq!(birth_proposal_priority(&bad), invalid, "{bad:?}");
    }
}

/// The per-token Eq. 4 advantage (flat − curved; positive ⇒ the curved atom is
/// cheaper) of ONE curved atom over `signal.len()` flat atoms on planted signal
/// columns, through the production scorer. Both arms reconstruct the same denoised
/// signal, so their residual spectra match, and with mutually orthogonal columns
/// the curved atom's code spectrum equals the flat atoms' pooled spectra.
fn eq4_flat_minus_curved(
    signal: &[Vec<f64>],
    n: usize,
    p: usize,
    g_dict: usize,
    basis_m: usize,
) -> f64 {
    use crate::eq4_description_length::eq4_fixed_distortion_description_length;
    use ndarray::Array2;

    let s = signal.len();
    let signal_mat =
        Array2::<f64>::from_shape_fn((n, p), |(i, j)| if j < s { signal[j][i] } else { 0.0 });
    let test_x = Array2::<f64>::from_shape_fn((n, p), |(i, j)| {
        signal_mat[[i, j]] + 0.1 * (0.7 * i as f64 + 1.9 * j as f64 + 0.3).sin()
    });
    let score = |gate: &Array2<f64>, dims: &[i64], columns: usize, curved: bool| {
        let contribution = signal_mat.clone();
        eq4_fixed_distortion_description_length(
            test_x.view(),
            signal_mat.view(),
            gate.view(),
            dims,
            (columns * p) as i64,
            n as i64,
            &[0.9],
            None,
            move |atom, take| {
                let mut out = Array2::<f64>::zeros((take.len(), p));
                for (out_row, &src) in take.iter().enumerate() {
                    if curved {
                        for col in 0..s {
                            out[[out_row, col]] = contribution[[src, col]];
                        }
                    } else {
                        out[[out_row, atom]] = contribution[[src, atom]];
                    }
                }
                Ok(out)
            },
        )
        .expect("Eq. 4 scoring succeeds")
    };
    let mut flat_gate = Array2::<f64>::zeros((n, g_dict));
    let mut flat_dims = vec![0_i64; g_dict];
    for atom in 0..s {
        flat_gate.column_mut(atom).fill(1.0);
        flat_dims[atom] = 1;
    }
    let mut curved_gate = Array2::<f64>::zeros((n, g_dict));
    curved_gate.column_mut(0).fill(1.0);
    let mut curved_dims = vec![0_i64; g_dict];
    curved_dims[0] = s as i64;
    let flat = score(&flat_gate, &flat_dims, s, false);
    let curved = score(&curved_gate, &curved_dims, basis_m, true);
    flat.per_target[0].bits - curved.per_target[0].bits
}

/// #2933 F22: the birth priority is not an Eq. 4 certificate. Candidates whose two
/// arms have MATCHED residual and code spectra are scored by the production Eq. 4
/// scorer and by the priority, and the signs disagree in BOTH directions:
///
/// * an anisotropic ellipse (variances 2 and 0.125, participation ratio ≈ 1.12)
///   charges a negative non-integer code coefficient the scorer never charges,
///   while Eq. 4 prefers the curved atom by its ≈ log₂((G−1)/2) support saving;
/// * four equal-energy harmonics (ŝ = 4, torus basis m = 25) are credited a code
///   saving `(ŝ−d−1)·½log₂(λ̂/δ)` the scorer does not see (both arms code the same
///   four variances), lifting the priority above zero while Eq. 4 prefers the flat
///   arm. At P = 108, N = 320 the per-token dictionary surcharge
///   `21·P·log₂N/(2N) = 29.49` bits sits between Eq. 4's support saving
///   `log₂C(G,4) − log₂G = 28.41` and the priority's `30.82`.
#[test]
fn birth_priority_is_not_an_eq4_certificate_2933() {
    use std::f64::consts::TAU;
    let topology = |span: f64| {
        let plan = crate::manifold::SaeAtomGeometryPlan::curved_prescreen_atom_for_span(span)
            .expect("the production pre-screen map must build its atom");
        (
            plan.intrinsic_dim(),
            plan.basis_size().expect("a built plan has a width"),
        )
    };

    // A negative priority for a candidate Eq. 4 prefers.
    let (n, p, g_dict) = (300usize, 6usize, 2048usize);
    let ellipse = vec![
        (0..n)
            .map(|i| 2.0 * (TAU * i as f64 / n as f64).cos())
            .collect::<Vec<f64>>(),
        (0..n)
            .map(|i| 0.5 * (TAU * i as f64 / n as f64).sin())
            .collect::<Vec<f64>>(),
    ];
    let variances = [2.0_f64, 0.125];
    let total: f64 = variances.iter().sum();
    let span = total * total / variances.iter().map(|v| v * v).sum::<f64>();
    let (d, m) = topology(span);
    let eq4_ellipse = eq4_flat_minus_curved(&ellipse, n, p, g_dict, m);
    let priority_ellipse = priority_bits(&BirthMdlPrescreen {
        rho: 1.0,
        span,
        intrinsic_dim: d,
        basis_size: m,
        signal_var: variances[0],
        noise_floor: 0.01,
        n_tokens: n as f64,
        p_out: p,
        g_dict,
        l0: 2.0,
    });
    assert!(
        eq4_ellipse > 1.0 && priority_ellipse < -(n as f64),
        "ellipse (span {span}): Eq. 4 advantage {eq4_ellipse} bits/token, priority \
         {priority_ellipse} bits"
    );

    // A positive priority for a candidate Eq. 4 rejects.
    let (n, p, g_dict) = (320usize, 108usize, 2048usize);
    let harmonics: Vec<Vec<f64>> = (1..=4)
        .map(|freq| {
            (0..n)
                .map(|i| 2.0 * (TAU * freq as f64 * i as f64 / n as f64).cos())
                .collect()
        })
        .collect();
    let (d, m) = topology(4.0);
    assert_eq!((d, m), (2, 25), "planted premise: the torus band");
    let eq4_harmonics = eq4_flat_minus_curved(&harmonics, n, p, g_dict, m);
    let priority_harmonics = priority_bits(&BirthMdlPrescreen {
        rho: 1.0,
        span: 4.0,
        intrinsic_dim: d,
        basis_size: m,
        signal_var: 2.0,
        noise_floor: 0.01,
        n_tokens: n as f64,
        p_out: p,
        g_dict,
        l0: 4.0,
    });
    assert!(
        eq4_harmonics < -0.5 && priority_harmonics > 0.5 * n as f64,
        "harmonics: Eq. 4 advantage {eq4_harmonics} bits/token, priority \
         {priority_harmonics} bits"
    );
}

/// #2933 F21: every report carries its score kind, and a model comparison across
/// kinds is refused unless the caller declares it heuristic.
#[test]
fn score_kinds_propagate_and_mixed_comparisons_are_refused_2933() {
    let codes = planted_codes(20, 4);
    let native = manifold_fit_description_length(
        &codes,
        &vec![vec![1.0, 0.5]; 4],
        0.2,
        0.8,
        &declared(50, 16.0),
    )
    .expect("the planted native ledger prices");
    assert_eq!(native.score_kind, DescriptionLengthScoreKind::GaussianSurrogate);
    let matched = matched_dl(3, 1, 8, 4.0, &[0.02; 10], 0.5);
    assert_eq!(
        matched.score_kind,
        DescriptionLengthScoreKind::HighResolutionIntrinsic
    );
    let test_x = ndarray::Array2::<f64>::from_shape_fn((8, 2), |(i, j)| {
        0.25 * (i * (j + 1)) as f64 + j as f64
    });
    let recon = test_x.mapv(|value| 0.8 * value);
    let gate = ndarray::Array2::<f64>::ones((8, 1));
    let contribution = recon.clone();
    let eq4 = crate::eq4_description_length::eq4_fixed_distortion_description_length(
        test_x.view(),
        recon.view(),
        gate.view(),
        &[1],
        4,
        4096,
        &[0.9],
        None,
        move |_, take| {
            let mut selected = ndarray::Array2::<f64>::zeros((take.len(), contribution.ncols()));
            for (out_row, &source_row) in take.iter().enumerate() {
                selected
                    .row_mut(out_row)
                    .assign(&contribution.row(source_row));
            }
            Ok(selected)
        },
    )
    .expect("the Eq. 4 fixture scores");
    assert_eq!(eq4.score_kind, DescriptionLengthScoreKind::GaussianSurrogate);
    assert_eq!(
        DescriptionLengthScoreKind::GaussianSurrogate.as_str(),
        "gaussian_surrogate"
    );
    assert_eq!(
        DescriptionLengthScoreKind::FiniteQuantizer.as_str(),
        "finite_quantizer"
    );

    let surrogate = ScoredBits {
        bits: native.total_bits,
        kind: native.score_kind,
    };
    let codebook = ScoredBits {
        bits: 17.0,
        kind: DescriptionLengthScoreKind::FiniteQuantizer,
    };
    let refused = description_length_delta(surrogate, codebook, ScoreComparison::SameKind);
    assert!(
        refused
            .as_ref()
            .is_err_and(|error| error.contains("gaussian_surrogate")
                && error.contains("finite_quantizer")),
        "{refused:?}"
    );
    assert_eq!(
        description_length_delta(surrogate, codebook, ScoreComparison::ExplicitHeuristic),
        Ok(native.total_bits - 17.0)
    );
    assert_eq!(
        description_length_delta(surrogate, surrogate, ScoreComparison::SameKind),
        Ok(0.0)
    );
}

/// #2933 F21: a covariance does not determine a rate. An equiprobable two-point
/// source `±σ` and a Gaussian of variance `σ²` receive the same surrogate rate at
/// every distortion, yet a one-bit sign codec round-trips the two-point source
/// exactly. At the surrogate's own one-bit operating point `D = σ²/4` the codec is
/// lossless, and at `D = 0` the surrogate rate is unbounded, so the surrogate is
/// neither a lower bound nor a message length for this source.
#[test]
fn gaussian_surrogate_is_not_a_bound_for_a_matched_covariance_discrete_source_2933() {
    let sigma = 1.7_f64;
    let source: Vec<f64> = (0..64)
        .map(|i| if i % 2 == 0 { sigma } else { -sigma })
        .collect();
    let n = source.len() as f64;
    let variance = source.iter().map(|x| x * x).sum::<f64>() / n;
    assert!((variance - sigma * sigma).abs() < 1.0e-12);
    // Encode one index bit per sample, decode it, and measure the reconstruction.
    let indices: Vec<bool> = source.iter().map(|&x| x >= 0.0).collect();
    let decoded: Vec<f64> = indices
        .iter()
        .map(|&positive| if positive { sigma } else { -sigma })
        .collect();
    let codec_bits_per_sample = indices.len() as f64 / n;
    let codec_distortion = source
        .iter()
        .zip(&decoded)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n;
    let (surrogate_one_bit, _) = reverse_water_filling(&[variance], variance / 4.0)
        .expect("a finite spectrum at a positive distortion");
    let (surrogate_lossless, _) = reverse_water_filling(&[variance], 0.0)
        .expect("zero distortion is a valid boundary");
    assert!((surrogate_one_bit - 1.0).abs() < 1.0e-12);
    assert!(surrogate_lossless.is_infinite());
    assert_eq!(codec_bits_per_sample, 1.0);
    assert_eq!(codec_distortion, 0.0);
    assert!(codec_distortion < variance / 4.0);
}

/// A periodic first-harmonic atom (basis `[1, sin 2πt, cos 2πt]`).
fn periodic_first_harmonic_plan() -> SaeAtomGeometryPlan {
    SaeAtomGeometryPlan::new(
        SaeAtomBasisKind::Periodic,
        1,
        SaeBasisResolution::PeriodicHarmonics { order: 1 },
        SaeReferenceMetricPlan::UnitCircle,
    )
    .expect("periodic first-harmonic plan")
}

#[test]
fn native_description_length_gains_nothing_by_moving_a_gate_below_a_threshold_2933_f15() {
    // The native kernel end to end. Atom 0's gates are multiplied by 1e-9 and its
    // decoder by 1e9, so every decoded product is unchanged while every gate of
    // atom 0 falls below 1e-8. The support, the occupancy, the coordinate codes, the
    // gate-amplitude code (its gate variance scales by 1e-18 and its decoded
    // sensitivity by 1e18) and the decoder-aware dictionary (its output sensitivity
    // scales by 1e-18 and its coefficient range by 1e9) are all unchanged, so the
    // whole message must cost the same number of bits.
    let n = 6;
    let plans = [periodic_first_harmonic_plan(), periodic_first_harmonic_plan()];
    let coords_0 = Array2::from_shape_vec((n, 1), vec![0.05, 0.2, 0.33, 0.5, 0.71, 0.9]).unwrap();
    let coords_1 = Array2::from_shape_vec((n, 1), vec![0.6, 0.1, 0.8, 0.45, 0.3, 0.15]).unwrap();
    let decoder_0 = Array2::from_shape_vec((3, 2), vec![1.0, -0.5, 0.4, 0.8, -0.3, 0.2]).unwrap();
    let decoder_1 = Array2::from_shape_vec((3, 2), vec![0.2, 0.9, -0.6, 0.1, 0.5, -0.7]).unwrap();
    let gates: Array2<f64> = Array2::from_shape_vec(
        (n, 2),
        vec![0.9, 0.2, 0.4, 0.7, 0.75, 0.1, 0.2, 0.55, 0.6, 0.35, 0.3, 0.8],
    )
    .unwrap();
    let scale = 1.0e9;
    let mut scaled_gates = gates.clone();
    scaled_gates.column_mut(0).mapv_inplace(|gate| gate / scale);
    let scaled_decoder_0 = decoder_0.mapv(|value| value * scale);
    assert!(scaled_gates.column(0).iter().all(|gate| gate.abs() < 1e-8));
    let original_decoded = crate::manifold::reconstruct_persisted_atom_set(
        &plans,
        &[decoder_0.view(), decoder_1.view()],
        &[coords_0.view(), coords_1.view()],
        gates.view(),
        2,
    )
    .unwrap();
    let rescaled_decoded = crate::manifold::reconstruct_persisted_atom_set(
        &plans,
        &[scaled_decoder_0.view(), decoder_1.view()],
        &[coords_0.view(), coords_1.view()],
        scaled_gates.view(),
        2,
    )
    .unwrap();
    for (left, right) in original_decoded.iter().zip(rescaled_decoded.iter()) {
        assert!((left - right).abs() <= 1e-12 * left.abs().max(1.0), "fixture decode differs");
    }

    // The fit leaves a residual; both representations reconstruct the same rows.
    let target = &original_decoded
        + &Array2::from_shape_fn(original_decoded.dim(), |(i, c)| {
            0.2 * (1.9 * (i + 1) as f64 + 0.8 * (c + 1) as f64).sin()
        });
    let describe = |gates: &Array2<f64>, decoder_0: &Array2<f64>, fitted: &Array2<f64>| {
        let decoders = [decoder_0.view(), decoder_1.view()];
        let coords = [coords_0.view(), coords_1.view()];
        let dictionary =
            persisted_decoder_dictionary_code(&plans, &decoders, &coords, gates.view(), None, 0)
                .expect("decoder dictionary code");
        native_manifold_description_length(NativeDescriptionLengthRequest {
            assignments: gates.view(),
            gate_model: NativeGateModel::Independent,
            geometry_plans: &plans,
            decoder_blocks: &decoders,
            coords: &coords,
            tier0_scale: None,
            target: target.view(),
            fitted: fitted.view(),
            dictionary: &dictionary,
        })
        .expect("native description length")
    };
    let original = describe(&gates, &decoder_0, &original_decoded);
    let rescaled = describe(&scaled_gates, &scaled_decoder_0, &rescaled_decoded);
    assert_eq!(
        rescaled.atom_occupancy,
        vec![1.0, 1.0],
        "a rescaled gate is still transmitted"
    );
    assert!(original.gate_amplitude_bits_per_token > 0.0, "the gates vary, so they cost bits");
    assert!(original.total_bits.is_finite() && original.total_bits > 0.0);
    for (ledger, left, right) in [
        ("selection", original.selection_bits, rescaled.selection_bits),
        ("coordinate", original.atom_code_bits_per_token[0], rescaled.atom_code_bits_per_token[0]),
        ("amplitude", original.gate_amplitude_bits_per_token, rescaled.gate_amplitude_bits_per_token),
        ("dictionary", original.dict_bits, rescaled.dict_bits),
        ("total", original.total_bits, rescaled.total_bits),
    ] {
        assert!(
            (left - right).abs() <= 1e-9 * left.abs().max(1.0),
            "moving a gate below a magnitude threshold changed the {ledger} bits: {left} vs {right}"
        );
    }
}

// Bug hunt — per-term summary EDF reads the WRONG penalty block when an
// UNPENALIZED random-effect main effect precedes the smooths in the design.
//
// Sibling of #1360 (per-term Wald test used a wrong covariance/penalty
// coordinate block). Here the defect is in the *penalty cursor* the model
// summary reconstructs to call `UnifiedFitResult::per_term_edf(coeff_range,
// penalty_cursor, k)`.
//
// The summary (both `src/main/model_summary.rs` and the Python-facing
// `crates/gam-pyffi/src/manifold_and_posterior_ffi.rs`) rebuilds the penalty
// cursor with:
//
//     penalty_cursor = 0
//     for (_name, _range) in design.random_effect_ranges { penalty_cursor += 1 }
//     for term in design.smooth.terms { ...per_term_edf(.., penalty_cursor, k); penalty_cursor += k }
//
// i.e. it assumes EVERY random-effect range owns exactly ONE penalty block.
// But the actual penalty layout built in `design_construction.rs` only emits a
// ridge block for a random effect when `spec.random_effect_terms[i].penalized`
// (line ~233: `if range.is_empty() || !...penalized { continue; }`).
//
// A factor-`by` smooth `s(x, by=g)` adds an UNPENALIZED treatment-coded
// random-effect main effect for `g` (term_builder.rs ~640, `penalized: false`)
// so that `g` appears in `random_effect_ranges` but contributes NO penalty
// block. The summary's cursor then over-counts by one and every following
// smooth term reads a penalty-block trace shifted by +1.
//
// This corrupts per-term EDF / ref_df / p-value whenever the influence matrix
// is unavailable so `per_term_edf` falls through to its
// `penalty_block_trace()[cursor..cursor+k]` path — which is the common
// production case, because column-conditioning drops the influence matrix
// (`penalty.rs:419 inf.coefficient_influence = None`).
//
// This test asserts the structural invariant directly on the built design: the
// number of leading random-effect ranges that the summary cursor SKIPS must
// equal the number of leading penalty blocks they actually own. With an
// unpenalized `by` factor present these disagree, so the reconstructed cursor
// for the first smooth term points past the smooth's own penalty block.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `y = sin(4x) + group offset + noise`, with a two-level factor `g` and an
/// independent covariate `z`. The formula `y ~ s(x, by=g) + s(z)` makes gam add
/// an UNPENALIZED treatment-coded main effect for the `by` factor `g`.
fn factor_by_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x: f64 = unit.sample(&mut rng);
            let z: f64 = unit.sample(&mut rng);
            let g = if i % 2 == 0 { "a" } else { "b" };
            let offset = if g == "a" { 0.0 } else { 0.7 };
            let y = (4.0_f64 * x).sin() + offset + noise.sample(&mut rng);
            StringRecord::from(vec![
                x.to_string(),
                z.to_string(),
                g.to_string(),
                y.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(
        ["x", "z", "g", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode")
}

#[test]
fn summary_penalty_cursor_matches_actual_penalty_layout() {
    init_parallelism();
    let data = factor_by_dataset(7, 400);
    let fit = fit_from_formula("y ~ s(x, by=g) + s(z)", &data, &FitConfig::default())
        .expect("fit y ~ s(x, by=g) + s(z)");

    let FitResult::Standard(std_fit) = &fit else {
        panic!("expected a standard Gaussian fit");
    };
    let design = &std_fit.design;

    // How many leading penalty blocks actually belong to random effects, read
    // straight from the built global penalty metadata. The global layout is
    // [linear ridge?, penalized-RE ridges, smooth penalties...]. None of the
    // formula-reachable linear terms are penalized, so the leading blocks here
    // are exactly the penalized-RE ridges.
    let re_names: std::collections::HashSet<&str> = design
        .random_effect_ranges
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    let leading_re_penalty_blocks = design
        .penaltyinfo
        .iter()
        .take_while(|info| {
            info.termname
                .as_deref()
                .map(|t| re_names.contains(t))
                .unwrap_or(false)
        })
        .count();

    // What the summary cursor must skip before the first smooth: the actual
    // leading non-smooth penalty blocks in the flat global penalty layout.
    // This intentionally differs from the old buggy reconstruction, which
    // advanced by one slot per random-effect range unconditionally.
    let summary_cursor_skips = design.leading_penalty_blocks_before_smooth();
    let buggy_cursor_skips = design.random_effect_ranges.len();

    // Sanity: the `by=g` factor really did introduce a random-effect range.
    assert!(
        !design.random_effect_ranges.is_empty(),
        "expected an unpenalized random-effect main effect for the by= factor; \
         random_effect_ranges was empty — formula plumbing changed"
    );
    assert_ne!(
        buggy_cursor_skips, leading_re_penalty_blocks,
        "regression fixture no longer contains an unpenalized random-effect \
         range; the old one-slot-per-range cursor would not desync"
    );

    // The invariant the summary RELIES ON: the number of random-effect ranges it
    // skips must equal the number of leading penalty blocks those ranges own,
    // not the number of random-effect coefficient ranges.
    assert_eq!(
        summary_cursor_skips, leading_re_penalty_blocks,
        "summary penalty-cursor desync: the summary advances the penalty cursor by \
         {summary_cursor_skips} before the first smooth, but \
         {leading_re_penalty_blocks} leading penalty blocks actually belong to \
         random effects. The first smooth's per_term_edf will read \
         penalty_block_trace[{summary_cursor_skips}..] instead of \
         [{leading_re_penalty_blocks}..], corrupting EDF / ref_df / p-value in the \
         influence-matrix-absent fallback path."
    );
}

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
// But the actual penalty layout only emits a penalty block for a random effect
// that `owns_penalty_block`: a one-level carrier of the model's level (the
// block that is the constant alone) contributes a column and NO penalty block.
//
// A factor-`by` smooth `s(x, by=g)` used to add an UNPENALIZED treatment-coded
// random-effect main effect for `g`, so that `g` appeared in
// `random_effect_ranges` but contributed NO penalty block. The summary's
// cursor then over-counted by one and every following smooth term read a
// penalty-block trace shifted by +1.
//
// The by= main effect is now a penalized full-level random block, so the
// formula fit no longer produces that range. The penalty-free random-effect
// range that remains is the one-level level carrier, so the fixture rebuilds
// the fitted design from the fit's own resolved spec with the `g` block turned
// into one, and checks the invariant on both designs.
//
// This corrupts per-term EDF / ref_df / p-value whenever the influence matrix
// is unavailable so `per_term_edf` falls through to its
// `penalty_block_trace()[cursor..cursor+k]` path — which is the common
// production case, because column-conditioning drops the influence matrix
// (`penalty.rs:419 inf.coefficient_influence = None`).
//
// This test asserts the structural invariant directly on the built design: the
// number of leading random-effect ranges that the summary cursor SKIPS must
// equal the number of leading penalty blocks they actually own. With a
// penalty-free carrier range present these disagree, so the reconstructed cursor
// for the first smooth term points past the smooth's own penalty block.

use csv::StringRecord;
use gam::smooth::{TermCollectionDesign, build_term_collection_design};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `y = sin(4x) + group offset + noise`, with a two-level factor `g` and an
/// independent covariate `z`. The formula `y ~ s(x, by=g) + s(z)` makes gam add
/// a random-effect main effect for the `by` factor `g`.
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

/// Checks the summary penalty cursor against `design`'s actual penalty layout
/// and walks every smooth term's penalty window from it. Returns the old
/// one-slot-per-range cursor and the number of leading penalty blocks the random
/// effects actually own.
fn assert_cursor_matches_layout(design: &TermCollectionDesign, label: &str) -> (usize, usize) {
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
    // leading non-smooth penalty blocks in the flat global penalty layout, read
    // via the same helper the fixed production summary / EDF / LR sites use.
    // This intentionally differs from the old buggy reconstruction, which
    // advanced by one slot per random-effect range unconditionally.
    let summary_cursor_skips = design.leading_penalty_blocks_before_smooth();
    let buggy_cursor_skips = design.random_effect_ranges.len();

    // Sanity: the `by=g` factor really did introduce a random-effect range.
    assert!(
        !design.random_effect_ranges.is_empty(),
        "{label}: expected a random-effect main effect for the by= factor; \
         random_effect_ranges was empty \u{2014} formula plumbing changed"
    );

    // The invariant the summary RELIES ON: the number of leading penalty blocks
    // the cursor skips must equal the number of leading penalty blocks the
    // random effects own, not the number of random-effect coefficient ranges.
    assert_eq!(
        summary_cursor_skips, leading_re_penalty_blocks,
        "{label}: summary penalty-cursor desync: the summary advances the penalty cursor by \
         {summary_cursor_skips} before the first smooth, but \
         {leading_re_penalty_blocks} leading penalty blocks actually belong to \
         random effects. The first smooth's per_term_edf will read \
         penalty_block_trace[{summary_cursor_skips}..] instead of \
         [{leading_re_penalty_blocks}..], corrupting EDF / ref_df / p-value in the \
         influence-matrix-absent fallback path."
    );

    // End-to-end: walking every smooth term's penalty window from that cursor
    // must consume the per-block penalty trace EXACTLY, never running past its
    // end. The old `random_effect_ranges.len()` seed overshoots here, so the
    // trailing smooth's `[cursor..cursor+k]` window would run off the end of the
    // per-block traces and `per_term_edf` would return 0 (#1883).
    let mut penalty_cursor = summary_cursor_skips;
    for term in &design.smooth.terms {
        let k = term.active_penalties.len();
        assert!(
            penalty_cursor + k <= design.penaltyinfo.len(),
            "{label}: smooth term '{}' penalty window [{penalty_cursor}..{}] runs past the \
             {} global penalty blocks \u{2014} penalty-cursor desync (#1883)",
            term.name,
            penalty_cursor + k,
            design.penaltyinfo.len()
        );
        penalty_cursor += k;
    }
    assert_eq!(
        penalty_cursor,
        design.penaltyinfo.len(),
        "{label}: the reconstructed penalty cursor must consume every global penalty block \
         exactly once; a leftover/overshoot means the per-term windows are \
         mis-aligned (#1883)"
    );
    (buggy_cursor_skips, leading_re_penalty_blocks)
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
    assert_cursor_matches_layout(&std_fit.design, "fitted design");

    // The same fit with the by= main effect turned into a one-level carrier of
    // the level: one column, the constant alone, which owns no penalty block.
    let mut carrier_spec = std_fit.resolvedspec.clone();
    let main_effect = carrier_spec
        .random_effect_terms
        .iter_mut()
        .find(|term| term.name == "g")
        .expect("the by= factor main effect `g` must be a random-effect term");
    let first_level = main_effect
        .frozen_levels
        .as_ref()
        .and_then(|levels| levels.first().copied())
        .expect("the fit freezes the by= factor's levels");
    main_effect.carries_level = true;
    main_effect.lenient_unseen = true;
    main_effect.frozen_levels = Some(vec![first_level]);
    let carrier_design = build_term_collection_design(data.values.view(), &carrier_spec)
        .expect("rebuild the design with a one-level carrier main effect");
    let (buggy_cursor_skips, leading_re_penalty_blocks) =
        assert_cursor_matches_layout(&carrier_design, "one-level carrier design");
    assert_ne!(
        buggy_cursor_skips, leading_re_penalty_blocks,
        "one-level carrier design: the carrier's range must add a column but no penalty \
         block, or the old one-slot-per-range cursor would not desync"
    );
}

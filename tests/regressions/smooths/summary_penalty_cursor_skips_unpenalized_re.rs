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

    // Reconstruct the penalty cursor EXACTLY as the FIXED model summary does
    // (`model_summary.rs` / `manifold_and_posterior_ffi.rs`): seed PAST any
    // leading shared `LinearTermRidge` block, then advance by ONE only for a
    // random-effect range that actually OWNS a penalty block — penalized AND
    // non-empty (`k_pen = penalized && !range.is_empty()`). The unpenalized
    // treatment-coded `by`-factor main effect owns no penalty block, so the
    // cursor must NOT advance for it (#1883). The earlier reconstruction used
    // `random_effect_ranges.len()`, advancing once per range unconditionally,
    // which over-counted here and slid every following smooth's penalty window.
    let linear_ridge_blocks = design
        .penaltyinfo
        .iter()
        .take_while(|info| {
            matches!(
                &info.penalty.source,
                gam::basis::PenaltySource::Other(s) if s == "LinearTermRidge"
            )
        })
        .count();
    let re_penalty_skips = design
        .random_effect_ranges
        .iter()
        .enumerate()
        .filter(|(re_idx, (_name, range))| {
            let penalized = std_fit
                .resolvedspec
                .random_effect_terms
                .get(*re_idx)
                .map(|t| t.penalized)
                .unwrap_or(true);
            penalized && !range.is_empty()
        })
        .count();

    // Sanity: the `by=g` factor really did introduce a random-effect range.
    assert!(
        !design.random_effect_ranges.is_empty(),
        "expected an unpenalized random-effect main effect for the by= factor; \
         random_effect_ranges was empty — formula plumbing changed"
    );

    // The invariant the FIXED summary establishes: the number of random-effect
    // ranges whose penalty block it actually skips must equal the number of
    // leading random-effect penalty blocks the design emitted. The unpenalized
    // `by`-factor range is present (sanity above) yet contributes ZERO to the
    // skip — which is the whole point of #1883. `random_effect_ranges.len()`
    // (the old reconstruction) would over-count here.
    assert_eq!(
        re_penalty_skips, leading_re_penalty_blocks,
        "penalty-cursor desync: the fixed reconstruction skips {re_penalty_skips} \
         random-effect penalty block(s), but the design's global penalty metadata \
         opens with {leading_re_penalty_blocks} leading random-effect block(s). \
         (random_effect_ranges.len() = {} includes the unpenalized by-factor main \
         effect, which owns no penalty block.)",
        design.random_effect_ranges.len()
    );

    // End-to-end: walking every smooth term's penalty window from that cursor
    // must consume the per-block penalty trace EXACTLY, never running past its
    // end. The old `random_effect_ranges.len()` seed overshoots here, so the
    // trailing smooth's `[cursor..cursor+k]` window would run off the end of the
    // per-block traces and `per_term_edf` would return 0 (#1883).
    let mut penalty_cursor = linear_ridge_blocks + re_penalty_skips;
    assert_eq!(
        penalty_cursor,
        linear_ridge_blocks + leading_re_penalty_blocks,
        "reconstructed cursor must land on the first smooth penalty block"
    );
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        assert!(
            penalty_cursor + k <= design.penaltyinfo.len(),
            "smooth term '{}' penalty window [{penalty_cursor}..{}] runs past the \
             {} global penalty blocks — penalty-cursor desync (#1883)",
            term.name,
            penalty_cursor + k,
            design.penaltyinfo.len()
        );
        penalty_cursor += k;
    }
    assert_eq!(
        penalty_cursor,
        design.penaltyinfo.len(),
        "the reconstructed penalty cursor must consume every global penalty block \
         exactly once; a leftover/overshoot means the per-term windows are \
         mis-aligned (#1883)"
    );
}

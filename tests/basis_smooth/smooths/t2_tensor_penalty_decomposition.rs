//! Regression for issue #1185: `t2(...)` must not be silently aliased to
//! `te(...)`.  `te(x, z)` emits one overlapping Kronecker-sum penalty per
//! margin.  `t2(x, z)` emits mgcv-style separable penalties for the marginal
//! penalized/null tensor subspaces.

use csv::StringRecord;
use gam::basis::PenaltySource;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

fn grid_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "z", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    let grid = 18usize;
    for i in 0..grid {
        for j in 0..grid {
            let x = i as f64 / (grid as f64 - 1.0);
            let z = j as f64 / (grid as f64 - 1.0);
            let y = (2.0 * x).sin() + (3.0 * z).cos() + x * z;
            rows.push(StringRecord::from(vec![
                x.to_string(),
                z.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode t2 grid dataset")
}

fn fit(formula: &str) -> StandardFitResult {
    init_parallelism();
    let data = grid_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &config)
        .unwrap_or_else(|err| panic!("{formula} should fit: {err:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for {formula}");
    };
    fit
}

#[test]
fn t2_uses_separable_penalty_decomposition_not_te_marginal_alias() {
    let te = fit("y ~ te(x, z, k=5)");
    let t2 = fit("y ~ t2(x, z, k=5)");

    let te_term = &te.design.smooth.terms[0];
    let t2_term = &t2.design.smooth.terms[0];

    // Classify by SOURCE rather than counting an undifferentiated total. Both
    // term types also carry one shared `TensorGlobalRidge` block, so a bare
    // `active_penalties.len()` no longer measures the decomposition: this test
    // read te as 3 and expected 2 purely because of that shared block, while
    // the #1185 structure it exists to guard was intact the whole time.
    fn classify(penalties: &[gam::basis::ActivePenalty]) -> (Vec<usize>, Vec<Vec<usize>>, usize) {
        let mut marginal: Vec<usize> = Vec::new();
        let mut separable: Vec<Vec<usize>> = Vec::new();
        let mut global_ridge = 0usize;
        for penalty in penalties {
            match &penalty.info.source {
                PenaltySource::TensorMarginal { dim, .. } => marginal.push(*dim),
                PenaltySource::TensorSeparable { penalized_margins } => {
                    separable.push(penalized_margins.clone())
                }
                PenaltySource::TensorGlobalRidge => global_ridge += 1,
                other => panic!("unexpected tensor penalty source: {other:?}"),
            }
        }
        marginal.sort_unstable();
        separable.sort();
        (marginal, separable, global_ridge)
    }

    let (te_marginal, te_separable, te_ridge) = classify(&te_term.active_penalties);
    let (t2_marginal, t2_separable, t2_ridge) = classify(&t2_term.active_penalties);

    // `te`: each margin's overlapping Kronecker-sum roughness, resolved into its
    // part through the other margin's null functions and its part through their
    // complement (#3951), and no separable block at all.
    assert_eq!(
        te_marginal,
        vec![0, 0, 1, 1],
        "te has two functional-ANOVA parts of each margin's roughness"
    );
    assert!(
        te_separable.is_empty(),
        "te must not emit separable tensor-subspace penalties: {te_separable:?}"
    );

    // `t2`: the marginal range/null tensor subspaces, and no marginal
    // Kronecker-sum block at all.
    assert_eq!(
        t2_separable,
        vec![vec![0], vec![0, 1], vec![1]],
        "t2 should split the coefficient space by marginal range/null tensor subspaces"
    );
    assert!(
        t2_marginal.is_empty(),
        "t2 must not emit marginal Kronecker-sum penalties: {t2_marginal:?}"
    );

    // The alias guard, stated on KIND rather than on a count. Comparing the two
    // lengths -- what this test used to do -- passes vacuously the moment the
    // two happen to agree, which is exactly the position `te` drifted into when
    // it picked up the shared ridge. Disjointness of the structural sources
    // cannot be satisfied by an aliased implementation at any count.
    assert!(
        te_separable.is_empty() && t2_marginal.is_empty(),
        "t2 must not be a te penalty alias: te={te_separable:?}, t2={t2_marginal:?}"
    );

    // Both decompositions share one joint polynomial null, `⊗_j null(S_j)`, and
    // #1561 gives each functional-ANOVA block of that null its own null-function
    // ridge (`tensor_null_function_block_ridges`). Cubic margins with
    // second-order penalties have a constant and a trend in each margin's null,
    // and the default sum-to-zero chart removes the grand constant, so exactly
    // three blocks survive: the x trend, the z trend and their interaction. Both
    // term types carry the same count, so neither accumulates extra shrinkage.
    assert_eq!(
        te_ridge, 3,
        "te carries one null-function ridge per retained ANOVA block: x trend, z trend, x·z trend"
    );
    assert_eq!(
        t2_ridge, te_ridge,
        "t2 shares te's joint polynomial null, so it carries the same null-function block ridges"
    );

    let t2_width = t2_term.coeff_range.len();
    assert!(
        t2_term.active_penalties.iter().all(|penalty| {
            penalty.matrix.nrows() == t2_width && penalty.matrix.ncols() == t2_width
        }),
        "every active t2 penalty must live in the transformed term coefficient space"
    );
}

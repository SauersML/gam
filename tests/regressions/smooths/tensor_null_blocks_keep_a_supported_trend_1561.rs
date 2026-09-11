// Issue #1561: the tensor double penalty shrank a supported trend together with
// absent ones.
//
// The joint null of `te(x, z)`'s roughness penalties is spanned by the constant,
// the x trend, the z trend and the x·z trend. Before this fix one ridge covered
// all of them with ONE smoothing parameter, and REML sizes a variance component
// by the average energy of the coefficients it governs. On the exported Poisson
// fixture (log mean 0.8 + 0.3 sin(x) + 0.2 z², a real x trend, no z or x·z trend)
// that parameter shrank the x trend to about a fifth of its unpenalized size, and
// the fit's error to the true mean was 0.2400 against 0.1671 with the ridge
// switched off.
//
// Each functional-ANOVA block of the null now carries its own REML coordinate,
// so `te(x, z)` shrinks its null trends the way `s(x) + s(z) + ti(x, z)` shrinks
// theirs. The reference-free statement of that is an ordering: on this fixture
// REML must keep the supported x trend (a small smoothing parameter) while it
// shrinks the absent z and x·z trends (large ones). A single shared parameter
// cannot express that ordering, and a fix that merely added coordinates without
// letting REML use them would fail it.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/measurements/issue_1561/poisson-tensor.csv"
);

#[test]
fn tensor_null_blocks_keep_a_supported_trend_and_shrink_absent_ones() {
    init_parallelism();
    let mut reader = csv::Reader::from_path(FIXTURE).expect("open the #1561 Poisson fixture");
    let rows: Vec<StringRecord> = reader
        .records()
        .map(|row| {
            let row = row.expect("fixture row");
            StringRecord::from(vec![row[0].to_string(), row[1].to_string(), row[2].to_string()])
        })
        .collect();
    assert_eq!(rows.len(), 300, "15 x 20 grid");
    let ds = encode_recordswith_inferred_schema(
        ["x", "z", "y"].into_iter().map(String::from).collect(),
        rows,
    )
    .expect("encode");
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=[6,6])", &ds, &cfg).expect("poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Poisson fit");
    };
    let log_lambdas: Vec<f64> = fit.fit.log_lambdas.iter().copied().collect();
    // Two margin roughness coordinates, then one null ridge per block retained
    // by the sum-to-zero chart, in block order: x trend, z trend, x·z trend.
    assert_eq!(
        log_lambdas.len(),
        5,
        "te(x, z) must carry its x, z and x·z null trends as separate REML coordinates; \
         log_lambdas = {log_lambdas:?}"
    );
    let (x_trend, z_trend, xz_trend) = (log_lambdas[2], log_lambdas[3], log_lambdas[4]);
    assert!(
        x_trend < z_trend && x_trend < xz_trend,
        "REML must keep the supported x trend and shrink the absent z and x·z trends: \
         log_lambdas = {log_lambdas:?}"
    );
}

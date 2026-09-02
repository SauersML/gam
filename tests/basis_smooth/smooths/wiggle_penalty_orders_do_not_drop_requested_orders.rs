use gam::families::wiggle::{WiggleBlockConfig, buildwiggle_block_input_from_orders, split_wiggle_penalty_orders};
use ndarray::array;

#[test]
fn wiggle_penalty_orders_requested_by_spec_are_not_silently_dropped() {
    let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let cfg = WiggleBlockConfig {
        degree: 3,
        num_internal_knots: 3,
        penalty_order: 2,
        double_penalty: false,
    };
    let requested_orders = vec![1, 3, 7];
    let (primary, extras) = split_wiggle_penalty_orders(cfg.penalty_order, &requested_orders)
        .expect("positive requested derivative orders must split");
    assert_eq!(primary, 1);
    assert_eq!(extras, vec![3, 7], "requested orders must be preserved");

    let mut effective_cfg = cfg.clone();
    effective_cfg.penalty_order = primary;

    let (primary_block, knots) = buildwiggle_block_input_from_seed(seed.view(), &effective_cfg)
        .expect("setup must build wiggle block");

    let baseline_penalty_count = primary_block.penalties.len();
    let supported_orders = [primary, extras[0]];
    let supported_block = buildwiggle_block_input_from_orders(
        seed.view(),
        &knots,
        cfg.degree,
        &supported_orders,
        cfg.double_penalty,
    )
    .expect("the supported order-three function penalty must be assembled");
    assert_eq!(
        supported_block.penalties.len(),
        baseline_penalty_count + 1,
        "every supported requested derivative order must materialize exactly one penalty"
    );

    let all_requested_orders = [primary, extras[0], extras[1]];
    // `expect_err` would require the Ok type to be `Debug` so it can print it,
    // and `ParameterBlockInput` is not (E0277). Match instead of deriving Debug
    // on a production type purely to satisfy a test formatter.
    let error = match buildwiggle_block_input_from_orders(
        seed.view(),
        &knots,
        cfg.degree,
        &all_requested_orders,
        cfg.double_penalty,
    ) {
        Ok(_) => panic!("an unavailable order-seven function derivative must be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        supported_block.penalties.len(),
        baseline_penalty_count + 1,
        "a rejected one-shot assembly must leave the previously built block untouched"
    );
    assert!(
        error.contains("Penalty order (7)"),
        "unsupported requested orders must be reported explicitly, not dropped: {error}"
    );
}

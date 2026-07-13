use gam::families::wiggle::{
    WiggleBlockConfig, append_selected_wiggle_function_penalties,
    buildwiggle_block_input_from_seed, split_wiggle_penalty_orders,
};
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

    let (mut block, knots) = buildwiggle_block_input_from_seed(seed.view(), &effective_cfg)
        .expect("setup must build wiggle block");

    let baseline_penalty_count = block.penalties.len();
    append_selected_wiggle_function_penalties(&mut block, &knots, cfg.degree, &extras[..1])
        .expect("the supported order-three function penalty must be appended");
    assert_eq!(
        block.penalties.len(),
        baseline_penalty_count + 1,
        "every supported requested derivative order must materialize exactly one penalty"
    );

    let error =
        append_selected_wiggle_function_penalties(&mut block, &knots, cfg.degree, &extras[1..])
            .expect_err("an unavailable order-seven function derivative must be rejected");
    assert_eq!(
        block.penalties.len(),
        baseline_penalty_count + 1,
        "a rejected derivative order must not mutate the penalty collection"
    );
    assert!(
        error.contains("Penalty order (7)"),
        "unsupported requested orders must be reported explicitly, not dropped: {error}"
    );
}

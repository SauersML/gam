use gam::families::wiggle::{
    WiggleBlockConfig, append_selected_wiggle_penalty_orders, buildwiggle_block_input_from_seed,
    split_wiggle_penalty_orders,
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
    let (primary, extras) = split_wiggle_penalty_orders(cfg.penalty_order, &requested_orders);

    let mut effective_cfg = cfg.clone();
    effective_cfg.penalty_order = primary;

    let (mut block, _knots) = buildwiggle_block_input_from_seed(seed.view(), &effective_cfg)
        .expect("setup must build wiggle block");

    let baseline_penalty_count = block.penalties.len();
    append_selected_wiggle_penalty_orders(&mut block, &extras)
        .expect("appending selected wiggle penalties must succeed");

    let appended_count = block.penalties.len() - baseline_penalty_count;
    assert_eq!(
        appended_count,
        extras.len(),
        "BUG: append_selected_wiggle_penalty_orders silently drops user-requested wiggle penalty orders when order >= basis dimension"
    );
}

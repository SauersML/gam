use gam::terms::smooth::{BlockwisePenalty, weighted_blockwise_penalty_sum};
use ndarray::array;

#[test]
#[should_panic(expected = "negative")]
fn bug_weighted_blockwise_penalty_sum_accepts_negative_weight_instead_of_erroring() {
    let penalties = vec![BlockwisePenalty::new(0..2, array![[2.0, 0.0], [0.0, 3.0]])];
    let _result = weighted_blockwise_penalty_sum(&penalties, &[-1.0], 2);
}

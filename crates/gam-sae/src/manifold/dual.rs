
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualKinkOp {
    Abs,
    Max,
    Min,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualKinkBranch {
    Left,
    Right,
    Tie,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualKinkBranchRecord {
    pub op: DualKinkOp,
    pub branch: DualKinkBranch,
    pub left_re: f64,
    pub right_re: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_records_tie_branch_without_hiding_the_kink() {
        let mut branches = Vec::new();
        let left = Dual::with_derivative(1.0, 2.0);
        let right = Dual::with_derivative(1.0, -3.0);

        let chosen = left.max_with_branch(right, &mut branches);

        assert_eq!(chosen, left);
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].op, DualKinkOp::Max);
        assert_eq!(branches[0].branch, DualKinkBranch::Tie);
        assert_eq!(branches[0].left_re, 1.0);
        assert_eq!(branches[0].right_re, 1.0);
    }

    #[test]
    fn min_records_tie_branch_without_hiding_the_kink() {
        let mut branches = Vec::new();
        let left = Dual::with_derivative(1.0, 2.0);
        let right = Dual::with_derivative(1.0, -3.0);

        let chosen = left.min_with_branch(right, &mut branches);

        assert_eq!(chosen, left);
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].op, DualKinkOp::Min);
        assert_eq!(branches[0].branch, DualKinkBranch::Tie);
    }

    #[test]
    fn abs_records_zero_branch_without_silently_selecting_a_side() {
        let mut branches = Vec::new();
        let dual = Dual::with_derivative(0.0, 7.0);

        let chosen = dual.abs_with_branch(&mut branches);

        assert_eq!(chosen.eps, 7.0);
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].op, DualKinkOp::Abs);
        assert_eq!(branches[0].branch, DualKinkBranch::Tie);
    }
}


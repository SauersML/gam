
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


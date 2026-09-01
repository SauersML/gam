use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub re: f64,
    pub eps: f64,
}

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

impl Dual {
    pub fn constant(re: f64) -> Self {
        Self { re, eps: 0.0 }
    }

    pub fn variable(re: f64) -> Self {
        Self { re, eps: 1.0 }
    }

    pub fn ln(self) -> Self {
        Self {
            re: self.re.ln(),
            eps: self.eps / self.re,
        }
    }

    pub fn sqrt(self) -> Self {
        let root = self.re.sqrt();
        Self {
            re: root,
            eps: self.eps / (2.0 * root),
        }
    }

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

impl Add for Dual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            eps: self.eps + rhs.eps,
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re + rhs,
            eps: self.eps,
        }
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            eps: self.eps - rhs.eps,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re - rhs,
            eps: self.eps,
        }
    }
}

impl Mul for Dual {
    type Output = Self;

}

impl Mul<f64> for Dual {
    type Output = Self;

}

impl Div for Dual {
    type Output = Self;

}

impl Div<f64> for Dual {
    type Output = Self;

}

impl Neg for Dual {
    type Output = Self;

}

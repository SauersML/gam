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

    pub fn with_eps(re: f64, eps: f64) -> Self {
        Self { re, eps }
    }

    pub fn with_derivative(re: f64, eps: f64) -> Self {
        Self { re, eps }
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

    pub fn recip(self) -> Self {
        Self {
            re: self.re.recip(),
            eps: -self.eps / (self.re * self.re),
        }
    }

    pub fn abs_with_branch(self, branches: &mut Vec<DualKinkBranchRecord>) -> Self {
        self.choose_max_with_branch(-self, DualKinkOp::Abs, branches)
    }

    pub fn max_with_branch(self, rhs: Self, branches: &mut Vec<DualKinkBranchRecord>) -> Self {
        self.choose_max_with_branch(rhs, DualKinkOp::Max, branches)
    }

    fn choose_max_with_branch(
        self,
        rhs: Self,
        op: DualKinkOp,
        branches: &mut Vec<DualKinkBranchRecord>,
    ) -> Self {
        let branch = if self.re > rhs.re {
            DualKinkBranch::Left
        } else if self.re < rhs.re {
            DualKinkBranch::Right
        } else {
            DualKinkBranch::Tie
        };
        branches.push(DualKinkBranchRecord {
            op,
            branch,
            left_re: self.re,
            right_re: rhs.re,
        });
        if matches!(branch, DualKinkBranch::Left | DualKinkBranch::Tie) {
            self
        } else {
            rhs
        }
    }

    pub fn min_with_branch(self, rhs: Self, branches: &mut Vec<DualKinkBranchRecord>) -> Self {
        let branch = if self.re < rhs.re {
            DualKinkBranch::Left
        } else if self.re > rhs.re {
            DualKinkBranch::Right
        } else {
            DualKinkBranch::Tie
        };
        branches.push(DualKinkBranchRecord {
            op: DualKinkOp::Min,
            branch,
            left_re: self.re,
            right_re: rhs.re,
        });
        if matches!(branch, DualKinkBranch::Left | DualKinkBranch::Tie) {
            self
        } else {
            rhs
        }
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

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re,
            eps: self.eps.mul_add(rhs.re, self.re * rhs.eps),
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re * rhs,
            eps: self.eps * rhs,
        }
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re / rhs,
            eps: self.eps / rhs,
        }
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

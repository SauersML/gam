use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub re: f64,
    pub eps: f64,
}

impl Dual {
    pub(crate) fn constant(re: f64) -> Self {
        Self { re, eps: 0.0 }
    }

    pub(crate) fn with_eps(re: f64, eps: f64) -> Self {
        Self { re, eps }
    }

    pub(crate) fn ln(self) -> Self {
        Self {
            re: self.re.ln(),
            eps: self.eps / self.re,
        }
    }

    pub(crate) fn sqrt(self) -> Self {
        let root = self.re.sqrt();
        Self {
            re: root,
            eps: self.eps / (2.0 * root),
        }
    }

    pub(crate) fn recip(self) -> Self {
        Self {
            re: self.re.recip(),
            eps: -self.eps / (self.re * self.re),
        }
    }

    pub(crate) fn abs(self) -> Self {
        self.max(-self)
    }

    pub(crate) fn max(self, rhs: Self) -> Self {
        if self.re >= rhs.re {
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

//! Exact continuous projection helpers for rank-1 periodic SAE charts.
//!
//! The manifold fit and transport diagnostics project ambient targets onto a
//! decoded periodic curve by enumerating every analytic stationary point.  The
//! torch trainer learns its coordinates through its nonlinear encoder and does
//! not use a separate coordinate E-step.

mod stationary_roots;

pub(crate) use stationary_roots::PeriodicCurveExtrema;

/// Periodic Fourier basis used by continuous chart transport diagnostics.
#[derive(Debug, Clone, Copy)]
pub enum ChartBasisKind {
    /// `[1, sin(2πt), cos(2πt), …, sin(2πHt), cos(2πHt)]`.
    Periodic { n_harmonics: usize },
}

impl ChartBasisKind {
    /// Number of basis columns produced by this basis.
    pub(crate) fn width(&self) -> usize {
        match self {
            ChartBasisKind::Periodic { n_harmonics } => 2 * n_harmonics + 1,
        }
    }

    /// Evaluate one basis row at the period-one coordinate `t`.
    pub(crate) fn eval_into(&self, t: f64, out: &mut [f64]) {
        match self {
            ChartBasisKind::Periodic { n_harmonics } => {
                out[0] = 1.0;
                for h in 1..=*n_harmonics {
                    let angle = std::f64::consts::TAU * h as f64 * t;
                    out[2 * h - 1] = angle.sin();
                    out[2 * h] = angle.cos();
                }
            }
        }
    }
}

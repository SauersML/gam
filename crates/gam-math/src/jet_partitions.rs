//! Bitmask-coefficient multi-directional jets used by marginal-slope and
//! latent-survival row kernels.
//!
//! The layout stores one coefficient per direction mask. The calculus itself
//! lives in [`crate::jet_algebra`]: this module only maps slot lists to masks.
use std::sync::atomic::{AtomicU64, Ordering};

pub static COMPOSE_UNARY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MUL_CALLS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub(crate) struct MultiDirJet {
    pub(crate) coeffs: Vec<f64>,
}

impl MultiDirJet {
    pub(crate) fn zero(n_dirs: usize) -> Self {
        Self {
            coeffs: vec![0.0; 1usize << n_dirs],
        }
    }

    pub(crate) fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    pub(crate) fn linear(n_dirs: usize, base: f64, first: &[f64]) -> Self {
        let mut out = Self::constant(n_dirs, base);
        for (idx, &value) in first.iter().take(n_dirs).enumerate() {
            out.coeffs[1usize << idx] = value;
        }
        out
    }

    pub(crate) fn with_coeffs(n_dirs: usize, coeffs: &[(usize, f64)]) -> Self {
        let mut out = Self::zero(n_dirs);
        for &(mask, value) in coeffs {
            if mask < out.coeffs.len() {
                out.coeffs[mask] = value;
            }
        }
        out
    }

    pub(crate) fn coeff(&self, mask: usize) -> f64 {
        self.coeffs[mask]
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        }
    }

    pub(crate) fn scale(&self, scalar: f64) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|value| scalar * value).collect(),
        }
    }

    pub(crate) fn mul(&self, other: &Self) -> Self {
        MUL_CALLS.fetch_add(1, Ordering::Relaxed);
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        for (mask, slot) in out.iter_mut().enumerate() {
            // The differentiation slots of coefficient `mask` are its set bits;
            // the shared Leibniz walker sums over subsets of those bits. A
            // slot-group (list of bit positions) maps back to a sub-mask, the
            // same submask enumeration the hand loop used — now one kernel
            // shared with `Tower4::mul` (#1151).
            let bits = bit_positions(mask);
            *slot = crate::jet_algebra::leibniz_product(
                bits.as_slice(),
                |t| self.coeffs[mask_of(t)],
                |c| other.coeffs[mask_of(c)],
            );
        }
        Self { coeffs: out }
    }

    pub(crate) fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        COMPOSE_UNARY_CALLS.fetch_add(1, Ordering::Relaxed);
        <Self as crate::jet_algebra::JetAlgebra<5>>::compose_unary(self, derivs)
    }
}

impl crate::jet_algebra::JetAlgebra<5> for MultiDirJet {
    #[inline]
    fn derivative(&self, slots: &[usize]) -> f64 {
        self.coeffs[mask_of(slots)]
    }

    fn map_derivatives<F>(&self, mut f: F) -> Self
    where
        F: FnMut(&[usize]) -> f64,
    {
        let mut out = vec![0.0; self.coeffs.len()];
        for (mask, value) in out.iter_mut().enumerate() {
            let bits = bit_positions(mask);
            *value = f(bits.as_slice());
        }
        Self { coeffs: out }
    }
}

/// The set-bit positions of `mask`, low to high — the differentiation slots of
/// that coefficient.
fn bit_positions(mask: usize) -> crate::jet_algebra::SlotBuf {
    let mut out = crate::jet_algebra::SlotBuf::new();
    let mut m = mask;
    while m != 0 {
        let bit = m.trailing_zeros() as usize;
        out.push_slot(bit);
        m &= m - 1;
    }
    out
}

/// Combine a slot-group (list of bit positions) back into a sub-mask.
fn mask_of(slots: &[usize]) -> usize {
    slots.iter().fold(0usize, |acc, &b| acc | (1usize << b))
}

// #932-2 cutover: `MultiDirJet::bilinear` (the 4-coeff `[base, d1, d2, d12]`
// constructor) and `MultiDirJet::sub` are consumed ONLY by the now test-only hand
// survival directional/bidirectional oracle (the production flex jet path uses the
// `flex_jet` runtime jet algebra, not `MultiDirJet`). They stay defined on the type
// but gated to the test build so the non-test lib does not carry them as dead code.
// (A child `#[cfg(test)] mod` can reach the private `coeffs` field, so the bodies are
// byte-identical to their former inherent-impl form.)
#[cfg(test)]
mod oracle_only_methods_tests {
    use super::*;

    impl MultiDirJet {
        pub(crate) fn bilinear(base: f64, d1: f64, d2: f64, d12: f64) -> Self {
            Self {
                coeffs: vec![base, d1, d2, d12],
            }
        }

        pub(crate) fn sub(&self, other: &Self) -> Self {
            Self {
                coeffs: self
                    .coeffs
                    .iter()
                    .zip(other.coeffs.iter())
                    .map(|(lhs, rhs)| lhs - rhs)
                    .collect(),
            }
        }
    }
}

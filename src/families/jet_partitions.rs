//! Shared, allocation-free lookup of bitmask set-partitions used by the
//! multi-directional jet `compose_unary` (Faà di Bruno) routines in the
//! marginal-slope and latent-survival families.
//!
//! Each entry `partitions(mask)` is the list of all set-partitions of the bits
//! of `mask`, with each partition represented as a `Vec<usize>` of disjoint
//! sub-masks whose bitwise OR equals `mask`. Total memory is bounded by the
//! Bell numbers up to `B(MAX_DIRS) = 4140` for `MAX_DIRS=8`. The 8-direction
//! cap covers the bernoulli marginal-slope rigid path's full uncontracted
//! fourth-tensor jet (`[e_q, e_g, e_q, e_g, e_q, e_g, e_q, e_g]`); no caller
//! currently needs more directions. Cache is computed lazily on first use.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub static COMPOSE_UNARY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MUL_CALLS: AtomicU64 = AtomicU64::new(0);
pub static ROW_NEGLOG_CALLS: AtomicU64 = AtomicU64::new(0);

const MAX_DIRS: usize = 8;
const TABLE_LEN: usize = 1usize << MAX_DIRS;

static CACHE: OnceLock<Vec<Vec<Vec<usize>>>> = OnceLock::new();

fn build_partitions(mask: usize) -> Vec<Vec<usize>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let first = mask & mask.wrapping_neg();
    let rest = mask ^ first;
    let mut out = Vec::new();
    let mut subset = rest;
    loop {
        let block = first | subset;
        for mut remainder in build_partitions(rest ^ subset) {
            remainder.push(block);
            out.push(remainder);
        }
        if subset == 0 {
            break;
        }
        subset = (subset - 1) & rest;
    }
    out
}

/// Returns the precomputed list of set-partitions for `mask`.
///
/// Supports masks in `0..=(1 << MAX_DIRS) - 1` (i.e. up to six-direction
/// jet machinery). Panics if `mask` is out of range.
pub fn partitions(mask: usize) -> &'static [Vec<usize>] {
    let table = CACHE.get_or_init(|| (0..TABLE_LEN).map(build_partitions).collect());
    &table[mask]
}

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

    pub(crate) fn bilinear(base: f64, d1: f64, d2: f64, d12: f64) -> Self {
        Self {
            coeffs: vec![base, d1, d2, d12],
        }
    }

    pub(crate) fn full_mask(&self) -> usize {
        self.coeffs.len() - 1
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
        for mask in 0..count {
            let mut total = 0.0;
            let mut submask = mask;
            loop {
                total += self.coeffs[submask] * other.coeffs[mask ^ submask];
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & mask;
            }
            out[mask] = total;
        }
        Self { coeffs: out }
    }

    pub(crate) fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        COMPOSE_UNARY_CALLS.fetch_add(1, Ordering::Relaxed);
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        out[0] = derivs[0];
        for (mask, value) in out.iter_mut().enumerate().skip(1) {
            let mut total = 0.0;
            for partition in partitions(mask) {
                let order = partition.len();
                if order == 0 || order >= derivs.len() {
                    continue;
                }
                let mut prod = 1.0;
                for &block in partition {
                    prod *= self.coeffs[block];
                }
                total += derivs[order] * prod;
            }
            *value = total;
        }
        Self { coeffs: out }
    }
}

//! Dense polynomial arithmetic for the survival marginal-slope closed-form
//! chain: scalar-coefficient ops over `PolyVec` and their multi-directional-jet
//! counterparts over `MultiDirJet`.
//!
//! Pure relocation from `survival_marginal_slope.rs` (issue #780
//! decomposition): `poly_mul`/`poly_sub`/`poly_add`/`poly_scale` over scalar
//! coefficient vectors, and `poly_add_jets`/`poly_scale_jets`/`poly_mul_jets`/
//! `poly_coeff_mask` over `MultiDirJet` coefficient vectors. These are
//! self-contained algebraic helpers; the entry points are re-imported by the
//! parent so every call site is unchanged. Bodies are byte-identical.

use super::PolyVec;
use crate::families::jet_partitions::MultiDirJet;
use smallvec::{SmallVec, smallvec};

#[inline]
pub(crate) fn poly_mul(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    if lhs.is_empty() || rhs.is_empty() {
        return PolyVec::new();
    }
    let n = lhs.len() + rhs.len() - 1;
    let mut out: PolyVec = smallvec![0.0; n];
    // Inner loop: out[i..i+rhs.len()] += lv * rhs. One sub-slicing bounds
    // check per outer iter lets LLVM elide the inner-loop bounds checks and
    // auto-vectorize the FMA. Hot path: ~13% self-time across poly_mul +
    // poly_add + poly_scale per the smoke-fixture profile.
    let out_slice = out.as_mut_slice();
    for (i, &lv) in lhs.iter().enumerate() {
        let dst = &mut out_slice[i..i + rhs.len()];
        for (d, &rv) in dst.iter_mut().zip(rhs.iter()) {
            *d += lv * rv;
        }
    }
    out
}

#[inline]
pub(crate) fn poly_sub(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    let mut out: PolyVec = SmallVec::new();
    out.extend_from_slice(lhs);
    if rhs.len() > lhs.len() {
        out.resize(rhs.len(), 0.0);
    }
    for (d, &v) in out[..rhs.len()].iter_mut().zip(rhs.iter()) {
        *d -= v;
    }
    out
}

#[inline]
pub(crate) fn poly_add(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    // Copy the longer operand verbatim, then add the shorter onto its
    // prefix. Avoids the redundant zero-fill of `smallvec![0.0; n]` plus
    // two additive passes from the legacy implementation.
    let (a, b) = if lhs.len() >= rhs.len() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let mut out: PolyVec = SmallVec::new();
    out.extend_from_slice(a);
    for (d, &v) in out[..b.len()].iter_mut().zip(b.iter()) {
        *d += v;
    }
    out
}

#[inline]
pub(crate) fn poly_scale(poly: &[f64], scale: f64) -> PolyVec {
    let mut out: PolyVec = SmallVec::with_capacity(poly.len());
    for &v in poly {
        out.push(scale * v);
    }
    out
}

pub(crate) fn poly_add_jets(lhs: &[MultiDirJet], rhs: &[MultiDirJet]) -> Vec<MultiDirJet> {
    let count = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let left = lhs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| MultiDirJet::zero(2));
        let right = rhs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| MultiDirJet::zero(2));
        out.push(left.add(&right));
    }
    out
}

pub(crate) fn poly_scale_jets(poly: &[MultiDirJet], scale: &MultiDirJet) -> Vec<MultiDirJet> {
    poly.iter().map(|coeff| coeff.mul(scale)).collect()
}

pub(crate) fn poly_mul_jets(lhs: &[MultiDirJet], rhs: &[MultiDirJet]) -> Vec<MultiDirJet> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let mut out = vec![MultiDirJet::zero(2); lhs.len() + rhs.len() - 1];
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            let prod = left.mul(right);
            out[i + j] = out[i + j].add(&prod);
        }
    }
    out
}

pub(crate) fn poly_coeff_mask(poly: &[MultiDirJet], mask: usize) -> Vec<f64> {
    poly.iter().map(|coeff| coeff.coeff(mask)).collect()
}

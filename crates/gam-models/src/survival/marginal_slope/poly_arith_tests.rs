#![cfg(test)]

//! Dense scalar polynomial arithmetic for the survival marginal-slope hand oracles.

use smallvec::{SmallVec, smallvec};

pub(super) type PolyVec = SmallVec<[f64; 32]>;

#[inline]
pub(super) fn poly_mul(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    if lhs.is_empty() || rhs.is_empty() {
        return PolyVec::new();
    }
    let n = lhs.len() + rhs.len() - 1;
    let mut out: PolyVec = smallvec![0.0; n];
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
pub(super) fn poly_sub(lhs: &[f64], rhs: &[f64]) -> PolyVec {
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
pub(super) fn poly_add(lhs: &[f64], rhs: &[f64]) -> PolyVec {
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
pub(super) fn poly_scale(poly: &[f64], scale: f64) -> PolyVec {
    let mut out: PolyVec = SmallVec::with_capacity(poly.len());
    for &v in poly {
        out.push(scale * v);
    }
    out
}

//! Dense polynomial arithmetic for the survival marginal-slope closed-form
//! chain: scalar-coefficient ops over `PolyVec`.
//!
//! `poly_mul`/`poly_sub`/`poly_add`/`poly_scale` over scalar coefficient vectors
//! (production, the grad-only first-order timepoint path). Their
//! `MultiDirJet`-coefficient counterparts (`poly_*_jets`/`poly_coeff_mask`) are now
//! test-only (the hand oracle) and live in `flex_oracle_structs_tests`, since the
//! #932-2 cutover routes the production flex jet path through the `flex_jet` runtime
//! jet algebra. Bodies are byte-identical to their pre-relocation form (#780).

use super::PolyVec;
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

// #932-2 cutover: the `MultiDirJet`-coefficient poly ops (`poly_add_jets` /
// `poly_scale_jets` / `poly_mul_jets` / `poly_coeff_mask`) feed ONLY the now
// test-only hand directional/bidirectional oracle — the production flex jet path
// uses the runtime `flex_jet` jet algebra, not `MultiDirJet`. Moved to the
// test-masked `flex_oracle_structs_tests` module so the non-test lib build does not
// carry them as dead code.

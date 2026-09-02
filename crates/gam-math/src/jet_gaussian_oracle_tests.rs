//! Gaussian location-scale Taylor-jet oracle (#932, FD-free exactness).
//!
//! Issue #932 builds out a generic, fixed-order truncated-Taylor jet algebra so
//! every family's [`super::jet_tower`]-derived
//! [`gam_models::row_kernel::RowKernel`]-shaped derivative tower (value / ∇ / H /
//! contracted-third / contracted-fourth) is MECHANICALLY derived from a single
//! row-NLL expression rather than hand-written. Deployment step 2 of the issue
//! elevates the jet-derived kernel to a UNIVERSAL ORACLE: an FD-free exactness
//! check that every hand-written kernel must match in CI, row by row, all
//! channels.
//!
//! The reusable oracle plumbing already lives in the tree —
//! [`crate::jet_tower::RowProgram`] (write the row NLL ONCE, generic
//! over the scalar), the production packed scalars
//! ([`crate::jet_scalar::Order2`] / [`crate::jet_scalar::OneSeed`] /
//! [`crate::jet_scalar::TwoSeed`]) that serve the `(v, g, H)` / contracted-third
//! / contracted-fourth channels, the dense [`crate::jet_tower::Tower4`], and the
//! channel-by-channel comparator [`crate::jet_tower::verify_kernel_channels`]
//! against the [`crate::jet_tower::KernelChannels`] a hand kernel claims.
//!
//! What this module adds is a NEW family wired to that universal oracle: the
//! Gaussian location-scale row negative log-likelihood — the canonical simplest
//! closed-form family the issue names first, and the one whose cross-block
//! (`∂η∂s`) curvature is the exact shape of the #736 sign-flip bug. The row loss
//! is written ONCE, generic over `S: JetScalar<2>` ([`GaussianLocScaleRow`]), and
//! the jet-derived tower — every channel, including the third / fourth
//! contractions realized through the production packed `OneSeed` / `TwoSeed`
//! scalars — is asserted equal, through the SAME `verify_kernel_channels`
//! comparator the production Bernoulli oracle uses, to an INDEPENDENT,
//! hand-derived closed-form Gaussian derivative tower
//! ([`gaussian_closed_form_channels`]) at several deterministic pseudo-random
//! points.
//!
//! This makes good on the issue's promise that "adding more families later is
//! straightforward": a new family is a `RowProgram` impl (one
//! expression) plus a `verify_kernel_channels` call against its own hand kernel —
//! no new algebra, no new comparator. The guard is genuine and distinct: the
//! comparand is external hand calculus for a real exponential family, so any
//! regression in an algebra primitive (a sign flip in the cross-Hessian, a
//! dropped Faà di Bruno term, an off-by-one in the `OneSeed`/`TwoSeed`
//! composition) is loud here even if it stayed self-consistent across the
//! packed/dense scalars.
//!
//! # The model
//!
//! Two primaries `p = (η, s)` with `s = log σ` (so the scale is unconstrained and
//! `e^{−2s} = 1/σ²` is the precision). With response `y` the Gaussian
//! location-scale row NLL (dropping the data-only `½ log 2π` constant, which only
//! shifts the value channel and leaves every derivative channel untouched) is
//!
//! ```text
//!   ℓ(η, s) = s + ½ e^{−2s} (y − η)².
//! ```
//!
//! Writing `r = y − η` and `w = e^{−2s}` the hand-derived tower is
//!
//! ```text
//!   ∂η ℓ      = −w r                       ∂s ℓ       = 1 − w r²
//!   ∂ηη ℓ     =  w                         ∂ss ℓ      = 2 w r²
//!   ∂ηs ℓ     = 2 w r                      (the #736 cross block)
//!   ∂ηηη = 0  ∂ηηs = −2w  ∂ηss = −4wr  ∂sss = −4wr²
//!   ∂ηηηη = 0 ∂ηηηs = 0   ∂ηηss = 4w   ∂ηsss = 8wr  ∂ssss = 8wr²
//! ```
//!
//! Every off-block (`∂η∂s`, `∂η∂η∂s`, …) is nonzero, so a dropped or sign-flipped
//! cross channel is caught by the oracle.

use crate::jet_scalar::JetScalar;
use crate::jet_tower::{KernelChannels, RowProgram};

/// One Gaussian location-scale fixture: the response `y` and the current
/// primaries `(η, s)` at which the row is linearized (`s = log σ`).
#[derive(Clone, Copy, Debug)]
struct GaussianRow {
    /// Observed continuous response `yᵢ`.
    y: f64,
    /// Location primary `ηᵢ` (the conditional mean).
    eta: f64,
    /// Log-scale primary `sᵢ = log σᵢ`.
    s: f64,
}

/// The Gaussian location-scale family, written ONCE as a generic
/// [`RowProgram<2>`] over the jet scalar `S`. The row NLL body uses
/// ONLY [`JetScalar`] ops (`sub`, `mul`, `scale`, `add`, `exp`); the per-row data
/// (`y`) enters as a plain `f64` constant — the single source of truth from which
/// every derivative channel is then exact by construction.
struct GaussianLocScaleRow {
    rows: Vec<GaussianRow>,
}

impl RowProgram<2> for GaussianLocScaleRow {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        let r = self
            .rows
            .get(row)
            .ok_or_else(|| format!("GaussianLocScaleRow: row {row} out of range"))?;
        // Primary order matches the closed form / channels: index 0 = η, 1 = s.
        Ok([r.eta, r.s])
    }

    fn eval<S: JetScalar<2>>(&self, row: usize, p: &[S; 2]) -> Result<S, String> {
        let data = self
            .rows
            .get(row)
            .ok_or_else(|| format!("GaussianLocScaleRow: row {row} out of range"))?;
        let eta = &p[0];
        let s = &p[1];
        // r = y − η  (a JetScalar; the y constant has all derivative channels 0).
        let r = S::constant(data.y).sub(eta);
        // w = e^{−2s}.
        let w = s.scale(-2.0).exp();
        // ℓ = s + ½ w r².  (No data-only normalizer added: this is the model NLL
        // up to the ½ log 2π constant the hand channels below also omit, so the
        // value channels match and every derivative channel is unaffected.)
        Ok(s.add(&w.mul(&r).mul(&r).scale(0.5)))
    }
}

/// INDEPENDENT hand-derived closed-form Gaussian location-scale channels at the
/// base point `(η, s)`, packaged as the [`KernelChannels`] a hand kernel would
/// claim. Derived by direct calculus (NOT via any jet) from
/// `ℓ = s + ½ e^{−2s} (y − η)²`, with `r = y − η`, `w = e^{−2s}`:
///
/// ```text
///   ℓ     = s + ½ w r²
///   ∇     = [−w r, 1 − w r²]
///   H     = [[ w,    2 w r ],
///            [ 2 w r, 2 w r²]]
///   ∂ηηη = 0  ∂ηηs = −2w  ∂ηss = −4wr  ∂sss = −4wr²
///   ∂ηηηη = 0 ∂ηηηs = 0   ∂ηηss = 4w   ∂ηsss = 8wr  ∂ssss = 8wr²
/// ```
///
/// The third / fourth tensors are contracted against the supplied directions so
/// the returned `KernelChannels` mirrors exactly what a hand kernel's
/// `row_third_contracted(dir)` / `row_fourth_contracted(u, v)` would return.
fn gaussian_closed_form_channels(
    row: &GaussianRow,
    third_dirs: &[[f64; 2]],
    fourth_pairs: &[([f64; 2], [f64; 2])],
) -> KernelChannels<2> {
    let r = row.y - row.eta;
    let w = (-2.0 * row.s).exp();

    let value = row.s + 0.5 * w * r * r;
    let gradient = [-w * r, 1.0 - w * r * r];
    let hessian = [[w, 2.0 * w * r], [2.0 * w * r, 2.0 * w * r * r]];

    // Symmetric third tensor by total order in s (index sum a+b+c).
    let t3 = |a: usize, b: usize, c: usize| -> f64 {
        match a + b + c {
            0 => 0.0,              // ∂ηηη
            1 => -2.0 * w,         // ∂ηηs
            2 => -4.0 * w * r,     // ∂ηss
            _ => -4.0 * w * r * r, // ∂sss
        }
    };
    // Symmetric fourth tensor by total order in s (index sum a+b+c+d).
    let t4 = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        match a + b + c + d {
            0 | 1 => 0.0,         // ∂ηηηη, ∂ηηηs
            2 => 4.0 * w,         // ∂ηηss
            3 => 8.0 * w * r,     // ∂ηsss
            _ => 8.0 * w * r * r, // ∂ssss
        }
    };

    let third = third_dirs
        .iter()
        .map(|dir| {
            let mut contracted = [[0.0_f64; 2]; 2];
            for a in 0..2 {
                for b in 0..2 {
                    let mut acc = 0.0;
                    for c in 0..2 {
                        acc += t3(a, b, c) * dir[c];
                    }
                    contracted[a][b] = acc;
                }
            }
            (*dir, contracted)
        })
        .collect();

    let fourth = fourth_pairs
        .iter()
        .map(|(u, v)| {
            let mut contracted = [[0.0_f64; 2]; 2];
            for a in 0..2 {
                for b in 0..2 {
                    let mut acc = 0.0;
                    for c in 0..2 {
                        for d in 0..2 {
                            acc += t4(a, b, c, d) * u[c] * v[d];
                        }
                    }
                    contracted[a][b] = acc;
                }
            }
            (*u, *v, contracted)
        })
        .collect();

    KernelChannels {
        value,
        gradient,
        hessian,
        third,
        fourth,
    }
}

/// A tiny deterministic LCG so the test points are pseudo-random yet fixed across
/// runs (NO `rand`, NO date/clock seeding — per the #932 rules).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}


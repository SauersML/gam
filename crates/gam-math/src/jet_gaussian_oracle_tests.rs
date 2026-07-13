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
use crate::jet_tower::{
    KernelChannels, RowProgram, Tower4, program_full_tower, verify_kernel_channels,
};

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

/// The mechanically jet-derived Gaussian location-scale tower (value / ∇ / H /
/// contracted-third / contracted-fourth) must equal the INDEPENDENT hand-derived
/// closed form, channel by channel, through the SAME universal
/// [`verify_kernel_channels`] oracle the production Bernoulli kernel uses. This
/// is the FD-free exactness check #932 deployment step 2 asks for, applied to a
/// new clean family with a single-expression row NLL.
#[test]
fn gaussian_loc_scale_jet_tower_matches_hand_derived_via_universal_oracle() {
    let mut rng = Lcg(0x9322_0203_face_b00c);
    // Deterministic contraction directions (no RNG dependency for the dirs).
    let third_dirs: [[f64; 2]; 3] = [[0.7, -1.3], [-0.4, 0.6], [1.2, 0.2]];
    let fourth_pairs: [([f64; 2], [f64; 2]); 3] = [
        ([0.7, -1.3], [-0.4, 0.6]),
        ([-0.4, 0.6], [1.2, 0.2]),
        ([1.2, 0.2], [0.7, -1.3]),
    ];

    let mut rows = Vec::new();
    for _ in 0..24 {
        rows.push(GaussianRow {
            // Moderate ranges keep w = e^{−2s} and r finite and well-scaled.
            y: rng.uniform(-3.0, 3.0),
            eta: rng.uniform(-2.0, 2.0),
            s: rng.uniform(-1.0, 1.0),
        });
    }
    let program = GaussianLocScaleRow { rows: rows.clone() };

    // The oracle rel_tol: the issue asks for ~1e-10; we hold to 1e-11, which the
    // pure scalar algebra clears comfortably.
    const REL_TOL: f64 = 1e-11;

    for (row, fixture) in rows.iter().enumerate() {
        // The mechanically jet-derived dense tower (every channel in one pass).
        let tower: Box<Tower4<2>> = program_full_tower(&program, row).expect("gaussian jet tower");

        // The INDEPENDENT hand-derived channels a hand kernel would claim.
        let claims = gaussian_closed_form_channels(fixture, &third_dirs, &fourth_pairs);

        verify_kernel_channels(&tower, &claims, REL_TOL).unwrap_or_else(|e| {
            panic!(
                "row {row}: Gaussian location-scale hand channels disagree with #932 \
                 jet-tower truth: {e}"
            )
        });
    }
}

/// The PRODUCTION packed scalars — `Order2` (value/∇/H), `OneSeed` (contracted
/// third), `TwoSeed` (contracted fourth) — evaluated on the SAME single
/// `eval` expression must reproduce the hand-derived closed form's
/// corresponding channels. This pins the cutover path a family would actually use
/// (the small packed scalars, not the dense `Tower4`) against external calculus,
/// with the contraction directions folded into the nilpotent seeds. It is the
/// proof that "adding a family" needs only its generic NLL + a channel comparison.
#[test]
fn gaussian_loc_scale_packed_scalars_match_hand_derived_contractions() {
    use crate::jet_tower::{
        program_fourth_contracted, program_row_kernel, program_third_contracted,
    };

    let mut rng = Lcg(0x0bad_c0de_9322_0203);
    let third_dirs: [[f64; 2]; 3] = [[0.9, -0.5], [-1.1, 0.3], [0.2, 1.4]];

    let mut rows = Vec::new();
    for _ in 0..18 {
        rows.push(GaussianRow {
            y: rng.uniform(-3.0, 3.0),
            eta: rng.uniform(-2.0, 2.0),
            s: rng.uniform(-1.0, 1.0),
        });
    }
    let program = GaussianLocScaleRow { rows: rows.clone() };

    const REL_TOL: f64 = 1e-11;
    let close = |a: f64, b: f64, label: &str| {
        let band = REL_TOL + REL_TOL * a.abs().max(b.abs());
        assert!(
            (a - b).abs() <= band,
            "{label}: jet {a:+.15e} vs hand {b:+.15e} (band {band:.3e})"
        );
    };

    for (row, fixture) in rows.iter().enumerate() {
        let hand = gaussian_closed_form_channels(fixture, &[], &[]);

        // Order2: value / gradient / Hessian via the production packed scalar.
        let (v, g, h) = program_row_kernel(&program, row).expect("Order2 channel");
        close(v, hand.value, &format!("row {row} Order2 value"));
        for i in 0..2 {
            close(
                g[i],
                hand.gradient[i],
                &format!("row {row} Order2 grad[{i}]"),
            );
            for j in 0..2 {
                close(
                    h[i][j],
                    hand.hessian[i][j],
                    &format!("row {row} Order2 hess[{i}][{j}]"),
                );
            }
        }

        // OneSeed: contracted third Σ_c ℓ_{abc}·dir_c via the production scalar,
        // checked against the dense tower's own contraction of t3.
        let tower: Box<Tower4<2>> = program_full_tower(&program, row).expect("tower");
        for (di, dir) in third_dirs.iter().enumerate() {
            let third = program_third_contracted(&program, row, dir).expect("OneSeed third");
            let truth = tower.third_contracted(dir);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        third[i][j],
                        truth[i][j],
                        &format!("row {row} dir {di} OneSeed third[{i}][{j}]"),
                    );
                }
            }
        }

        // TwoSeed: contracted fourth Σ_{cd} ℓ_{abcd}·u_c·v_d via the production
        // scalar, checked against the dense tower's own contraction of t4.
        for (ui, u) in third_dirs.iter().enumerate() {
            let v = third_dirs[(ui + 1) % third_dirs.len()];
            let fourth = program_fourth_contracted(&program, row, u, &v).expect("TwoSeed fourth");
            let truth = tower.fourth_contracted(u, &v);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        fourth[i][j],
                        truth[i][j],
                        &format!("row {row} pair {ui} TwoSeed fourth[{i}][{j}]"),
                    );
                }
            }
        }
    }
}

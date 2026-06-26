//! Per-fit first-order optimality certificate for the SAE LAML criterion
//! (issue #934).
//!
//! # What this is
//!
//! The recurring structural bug genus in this engine is *objective↔gradient
//! desync*: the criterion value `V(ρ)` and its analytic derivative `∇V(ρ)`
//! are computed by separate code paths that drift apart (#752, #748, #808,
//! #901, …). Every one of those bugs was eventually diagnosed by a human
//! running one finite-difference comparison by hand at the returned optimum:
//! the analytic gradient claimed convergence while a finite difference of the
//! *actual objective* disagreed. The engine never ran that check on itself.
//!
//! This module makes the engine run it on every real fit. Once, at the
//! converged outer optimum `ρ̂`, outside all hot loops:
//!
//! 1. Draw one deterministic direction `v` on the ρ-sphere from the problem
//!    fingerprint (no `Date`/random-source nondeterminism).
//! 2. Central-difference the criterion **value path** at `ρ̂ ± h v` with a
//!    Richardson second step `2h` to estimate the FD's own error bar.
//! 3. Compare against the analytic directional derivative `∇V(ρ̂)·v`.
//! 4. Record a [`CriterionCertificate`] on the fit payload.
//!
//! # Why finite differences are legal here
//!
//! The exact-REML-only policy bans approximate quantities from *producing*
//! the fit. This FD probe does not produce anything the fit consumes — it
//! **audits** the production analytic gradient against the production value
//! path, at a single point, after convergence. FD is the audit instrument,
//! not the estimator. That boundary is what makes it the runtime enforcement
//! layer for the criterion-atom architecture (#931): atoms make desync
//! structurally hard to write; this certificate makes any residue observable
//! in production, where the real data shapes that trigger #901-class desyncs
//! actually occur. It is the same relationship theta-correction atoms have to
//! their invariants — the invariant is enforced structurally and audited at
//! runtime.
//!
//! # Cost
//!
//! Two value-path evaluations for the central difference plus two for the
//! Richardson step: four criterion evaluations at the single final point,
//! seconds even at biobank scale. The value path is evaluated **without**
//! warm-state shortcuts that would alias it to the gradient path — that
//! aliasing is exactly what must be audited — so the probe is taken on a
//! clone of the pristine baseline term whose caches start cold and naturally
//! miss the gradient path's converged state.

use ndarray::ArrayView1;

/// The result of the first-order self-audit at the converged outer optimum.
///
/// Disagreement does not, by itself, fail the fit — it *names the broken term
/// loudly* in the result and the report, converting the next desync from a
/// multi-week biobank-stall investigation into a one-line diagnosis at the
/// moment of introduction. Consumers decide the policy verdict from
/// [`Self::agreement_rel`] against their own tolerance.
#[derive(Debug, Clone, Copy)]
pub struct CriterionCertificate {
    /// `‖∇V(ρ̂)‖₂`, the analytic gradient norm reported as converged.
    pub grad_norm: f64,
    /// Central-difference directional derivative of the **value path**:
    /// `[V(ρ̂ + h v) − V(ρ̂ − h v)] / (2h)`.
    // FD-OK: FD-audit certificate oracle field verifying the analytic directional derivative
    pub fd_directional: f64, // fd-ok: FD-audit certificate, not in math path
    // END-FD-OK
    /// Analytic directional derivative `∇V(ρ̂)·v` from the production gradient
    /// path, on the same unit direction `v`.
    pub analytic_directional: f64,
    /// Richardson error-bar estimate of the finite difference itself:
    /// `|D(h) − D(2h)| / 3`, the leading `O(h²)` truncation term of the central
    /// difference. The FD/analytic gap is only meaningful relative to this — a
    /// gap below the error bar is consistent with exact agreement.
    // FD-OK: Richardson error-bar of the FD-audit oracle (reporting only)
    pub fd_error_bar: f64, // fd-ok: FD-audit certificate, not in math path
    // END-FD-OK
    /// The probe step `h` actually used (scaled to the coordinate magnitude).
    pub step: f64,
    /// Whether the criterion's curvature at `ρ̂` is usable: the value-path
    /// evaluations all returned finite, an undamped factorization succeeded.
    /// `false` flags a railed/degenerate optimum (the #748 indefinite-`H+Sλ`
    /// signature) even when the directional probe itself agrees.
    pub well_posed: bool,
}

impl CriterionCertificate {
    /// Signed FD−analytic disagreement, normalized to the larger of the two
    /// directional magnitudes (so a flat direction with both near zero does
    /// not manufacture a spurious relative blow-up) and floored by the FD
    /// error bar (so we never claim a disagreement the probe itself cannot
    /// resolve).
    #[must_use]
    pub fn agreement_rel(&self) -> f64 {
        // FD-OK: comparing FD-audit oracle against the analytic directional derivative
        let scale = self
            .analytic_directional
            .abs()
            .max(self.fd_directional.abs()) // fd-ok: FD-audit certificate, not in math path
            .max(self.fd_error_bar) // fd-ok: FD-audit certificate, not in math path
            .max(1e-12);
        (self.fd_directional - self.analytic_directional).abs() / scale // fd-ok: FD-audit certificate, not in math path
        // END-FD-OK
    }

    /// The certificate's verdict against a relative tolerance: `true` means the
    /// analytic gradient and the value-path FD agree at the optimum to within
    /// `rel_tol` AND the curvature is well-posed. `false` is the loud
    /// desync/rail flag — not necessarily a failed fit, but a named one.
    #[must_use]
    pub fn passes(&self, rel_tol: f64) -> bool {
        self.well_posed && self.agreement_rel() <= rel_tol
    }
}

/// A deterministic unit direction on the ρ-sphere derived purely from the
/// problem fingerprint `(ρ̂ values, dimension)`.
///
/// No `Date`, no thread-RNG, no global entropy: the same fit fingerprint
/// always yields the same probe direction, so the certificate is reproducible
/// and CI-stable. A SplitMix64 hash of each coordinate index mixed with the
/// fingerprint seed produces a fixed pseudo-random direction; it is then
/// normalized. If every coordinate hashes to zero (impossible for a nonempty
/// vector with this mixer) the function falls back to the first axis.
#[must_use]
pub fn deterministic_probe_direction(rho_hat: ArrayView1<'_, f64>) -> Vec<f64> {
    let n = rho_hat.len();
    if n == 0 {
        return Vec::new();
    }
    // Fold the optimum coordinates into a 64-bit seed (finite-safe: NaN/Inf
    // bit patterns are tolerated, they only perturb the seed deterministically).
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for (idx, &value) in rho_hat.iter().enumerate() {
        seed =
            splitmix64(seed ^ value.to_bits() ^ (idx as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
    }
    let mut dir = vec![0.0_f64; n];
    let mut s = seed;
    let mut norm_sq = 0.0_f64;
    for slot in dir.iter_mut() {
        s = splitmix64(s);
        // Map the hashed bits to a symmetric (−1, 1) coordinate.
        let unit = (s >> 11) as f64 / ((1u64 << 53) as f64); // [0, 1)
        let coord = 2.0 * unit - 1.0;
        *slot = coord;
        norm_sq += coord * coord;
    }
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for slot in dir.iter_mut() {
            *slot /= norm;
        }
    } else {
        dir[0] = 1.0;
    }
    dir
}

/// SplitMix64 — a tiny, well-distributed integer mixer used only to make the
/// probe direction a deterministic function of the fit fingerprint. Thin
/// wrapper over the canonical implementation in
/// [`gam_linalg::utils::splitmix64_hash`].
fn splitmix64(state: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(state)
}

/// The probe step `h` scaled to the magnitude of the optimum: log-ρ
/// coordinates are `O(1)`–`O(10)`, so a relative step keeps the central
/// difference inside the smooth quadratic region while staying well above
/// double-precision round-off of the criterion (~`1e-8` relative).
#[must_use]
pub fn probe_step(rho_hat: ArrayView1<'_, f64>) -> f64 {
    const BASE: f64 = 1e-4;
    let scale = rho_hat.iter().fold(1.0_f64, |m, &x| m.max(x.abs()));
    BASE * scale
}

/// Per-coordinate probe step for a single outer-ρ axis.
///
/// [`probe_step`] collapses the whole vector to one global step
/// (`1e-4 · max_i|ρ_i|`), which under-resolves a small coordinate whenever some
/// other axis is large. When differencing one axis at a time, scale the step to
/// that coordinate's own magnitude, with a unit floor so a near-zero coordinate
/// still gets a usable step.
pub fn probe_step_for(rho_i: f64) -> f64 {
    const BASE: f64 = 1e-4;
    BASE * rho_i.abs().max(1.0)
}

/// Samples of the criterion **value path** taken at the four probe points
/// around `ρ̂` along the unit direction `v`, plus the analytic directional
/// derivative — everything the certificate needs, with no dependence on the
/// SAE term type so this is unit-testable in isolation.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalSamples {
    /// `V(ρ̂ + h v)`.
    pub plus_h: f64,
    /// `V(ρ̂ − h v)`.
    pub minus_h: f64,
    /// `V(ρ̂ + 2h v)`.
    pub plus_2h: f64,
    /// `V(ρ̂ − 2h v)`.
    pub minus_2h: f64,
    /// The step `h`.
    pub step: f64,
    /// `‖∇V(ρ̂)‖₂`.
    pub grad_norm: f64,
    /// `∇V(ρ̂)·v`.
    pub analytic_directional: f64,
    /// Whether every value-path evaluation returned a finite criterion at a
    /// well-conditioned (undamped-factorable) inner optimum.
    pub well_posed: bool,
}

/// Assemble the [`CriterionCertificate`] from directional value samples.
///
/// The central difference at step `h` is `D(h) = (V₊ − V₋) / 2h`; at step
/// `2h` it is `D(2h) = (V₊₊ − V₋₋) / 4h`. For a smooth criterion both
/// approximate `∇V·v` with leading error `(h²/6)·V‴·v` and `(4h²/6)·V‴·v`, so
/// `|D(h) − D(2h)| = h²/2·|V‴·v| + O(h⁴)` and the Richardson-extrapolated FD
/// error bar of `D(h)` is `|D(h) − D(2h)| / 3` (the standard central-difference
/// Richardson remainder). The reported `fd_directional` is `D(h)`.
#[must_use]
pub fn certificate_from_samples(s: &DirectionalSamples) -> CriterionCertificate {
    // FD-OK: Richardson FD oracle constructed to audit the analytic directional derivative
    let d_h = (s.plus_h - s.minus_h) / (2.0 * s.step);
    let d_2h = (s.plus_2h - s.minus_2h) / (4.0 * s.step);
    let fd_error_bar = (d_h - d_2h).abs() / 3.0; // fd-ok: FD-audit certificate, not in math path
    CriterionCertificate {
        grad_norm: s.grad_norm,
        fd_directional: d_h, // fd-ok: FD-audit certificate, not in math path
        analytic_directional: s.analytic_directional,
        fd_error_bar, // fd-ok: FD-audit certificate, not in math path
        // END-FD-OK
        step: s.step,
        well_posed: s.well_posed
            && s.plus_h.is_finite()
            && s.minus_h.is_finite()
            && s.plus_2h.is_finite()
            && s.minus_2h.is_finite(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// A quadratic `V(ρ) = ½ ρᵀ A ρ + bᵀρ` has exact analytic directional
    /// derivative and a central difference that recovers it to machine
    /// precision (third derivative is zero), so the certificate must pass at a
    /// vanishing relative gap and a vanishing error bar.
    #[test]
    fn quadratic_certificate_agrees_exactly() {
        // V(ρ) = ½(2ρ₀² + 3ρ₁²) + (ρ₀ − 2ρ₁); ∇V = (2ρ₀+1, 3ρ₁−2).
        let v = |r: &[f64]| 0.5 * (2.0 * r[0] * r[0] + 3.0 * r[1] * r[1]) + (r[0] - 2.0 * r[1]);
        let rho = Array1::from(vec![0.7_f64, -1.3]);
        let grad = [2.0 * rho[0] + 1.0, 3.0 * rho[1] - 2.0];
        let dir = deterministic_probe_direction(rho.view());
        let h = probe_step(rho.view());
        let at = |sign: f64, mult: f64| {
            let p: Vec<f64> = (0..2).map(|i| rho[i] + sign * mult * h * dir[i]).collect();
            v(&p)
        };
        let grad_norm = (grad[0] * grad[0] + grad[1] * grad[1]).sqrt();
        let analytic_directional = grad[0] * dir[0] + grad[1] * dir[1];
        let samples = DirectionalSamples {
            plus_h: at(1.0, 1.0),
            minus_h: at(-1.0, 1.0),
            plus_2h: at(1.0, 2.0),
            minus_2h: at(-1.0, 2.0),
            step: h,
            grad_norm,
            analytic_directional,
            well_posed: true,
        };
        let cert = certificate_from_samples(&samples);
        assert!(
            cert.agreement_rel() < 1e-6,
            "quadratic FD must match analytic: rel {}, fd {}, analytic {}",
            cert.agreement_rel(),
            cert.fd_directional, // fd-ok: FD-audit certificate, not in math path
            cert.analytic_directional
        );
        assert!(
            cert.fd_error_bar < 1e-6, // fd-ok: FD-audit certificate, not in math path
            "quadratic has zero third derivative, error bar must be tiny: {}",
            cert.fd_error_bar // fd-ok: FD-audit certificate, not in math path
        );
        assert!(cert.passes(1e-4), "well-posed quadratic must certify");
    }

    /// A planted desync — the analytic directional derivative is deliberately
    /// off by 30% from the true (value-path) slope — must be caught loudly: the
    /// relative agreement blows past any sane tolerance even though the value
    /// path itself is perfectly smooth.
    #[test]
    fn planted_desync_is_caught() {
        let v = |r: &[f64]| r[0].sin() + 0.5 * r[1] * r[1];
        let rho = Array1::from(vec![0.4_f64, 0.9]);
        let true_grad = [rho[0].cos(), rho[1]];
        // Desynced analytic gradient: 30% too large in coord 0.
        let bad_grad = [1.3 * true_grad[0], true_grad[1]];
        let dir = deterministic_probe_direction(rho.view());
        let h = probe_step(rho.view());
        let at = |sign: f64, mult: f64| {
            let p: Vec<f64> = (0..2).map(|i| rho[i] + sign * mult * h * dir[i]).collect();
            v(&p)
        };
        let grad_norm = (bad_grad[0] * bad_grad[0] + bad_grad[1] * bad_grad[1]).sqrt();
        let analytic_directional = bad_grad[0] * dir[0] + bad_grad[1] * dir[1];
        let samples = DirectionalSamples {
            plus_h: at(1.0, 1.0),
            minus_h: at(-1.0, 1.0),
            plus_2h: at(1.0, 2.0),
            minus_2h: at(-1.0, 2.0),
            step: h,
            grad_norm,
            analytic_directional,
            well_posed: true,
        };
        let cert = certificate_from_samples(&samples);
        assert!(
            !cert.passes(1e-3),
            "30% desync must fail the certificate: rel {}, fd {}, analytic {}",
            cert.agreement_rel(),
            cert.fd_directional, // fd-ok: FD-audit certificate, not in math path
            cert.analytic_directional
        );
    }

    /// The probe direction is deterministic in the fingerprint and is a unit
    /// vector.
    #[test]
    fn probe_direction_is_deterministic_unit() {
        let rho = Array1::from(vec![1.0_f64, -2.0, 0.5, 3.3]);
        let a = deterministic_probe_direction(rho.view());
        let b = deterministic_probe_direction(rho.view());
        assert_eq!(a, b, "same fingerprint must give same direction");
        let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "direction must be unit, got {norm}"
        );
        // A different optimum gives a different direction.
        let rho2 = Array1::from(vec![1.0_f64, -2.0, 0.5, 3.4]);
        let c = deterministic_probe_direction(rho2.view());
        assert_ne!(a, c, "different fingerprint must give different direction");
    }

    /// A non-finite value-path sample marks the certificate not-well-posed even
    /// when the recorded directional numbers happen to agree.
    #[test]
    fn nonfinite_sample_marks_not_well_posed() {
        let samples = DirectionalSamples {
            plus_h: f64::NAN,
            minus_h: 1.0,
            plus_2h: 2.0,
            minus_2h: 0.0,
            step: 1e-4,
            grad_norm: 1.0,
            analytic_directional: 0.0,
            well_posed: true,
        };
        let cert = certificate_from_samples(&samples);
        assert!(
            !cert.well_posed,
            "NaN value sample must flag not-well-posed"
        );
        assert!(!cert.passes(1.0), "not-well-posed never certifies");
    }

    /// A non-finite Richardson sample (`plus_2h` or `minus_2h`) also marks
    /// the certificate not-well-posed — the error bar is meaningless if those
    /// evaluations failed, even when the ±h samples are finite.
    #[test]
    fn nonfinite_richardson_sample_marks_not_well_posed() {
        let samples = DirectionalSamples {
            plus_h: 1.1,
            minus_h: 0.9,
            plus_2h: f64::NAN,
            minus_2h: 0.8,
            step: 1e-4,
            grad_norm: 1.0,
            analytic_directional: 1.0,
            well_posed: true,
        };
        let cert = certificate_from_samples(&samples);
        assert!(
            !cert.well_posed,
            "NaN Richardson sample must flag not-well-posed"
        );
        assert!(!cert.passes(1.0), "not-well-posed never certifies");
    }
}

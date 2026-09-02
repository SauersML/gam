//! #2023 C4 — Tier-0 shared mean (the manifold-tier analogue of
//! [`crate::tiered::Tier0Mean`]) tests: the shared mean de-means the target and
//! reconstructs exactly (round-trip), and — the headline — moving the global DC
//! into Tier-0 makes a DC-constant "zombie" atom EV-invisible BY CONSTRUCTION, so
//! the 6-circle fixture has ZERO zero-decoder survivors.
//!
//! A DC-constant zombie is an atom whose decoder loads ONLY the constant basis
//! column (a pure per-row constant, no manifold structure). Without Tier-0 the
//! atoms fit the RAW target, whose column mean is the global DC, so the zombie's
//! constant column loads that mean and the atom "survives" selection by carrying a
//! slice of it (the co-collapse-to-mean class, #10/#1893): removing it drops
//! explained variance, so its leave-one-atom-out ΔEV is positive and it is kept and
//! PC-reseeded. With the shared mean carried by Tier-0 the atoms fit the DE-MEANED
//! target `Z − μ`, whose column mean is zero, so the zombie's constant column loads
//! NOTHING — it decodes ≈0, earns essentially no explained variance, and its ΔEV is
//! non-positive: it is NOT a survivor. Genuine curved atoms (the 6 circles) earn
//! positive ΔEV in BOTH modes: Tier-0 removes the mean, not the structure. (The
//! failure mode Tier-0 exists to prevent — installing μ on top of a dictionary that
//! ALSO fit the raw mean into a decoder — is the DOUBLE-SUBTRACTION HAZARD below: it
//! biases every reconstruction by `+μ` and corrupts every atom's ΔEV, so the fixture
//! must fit the zombie against the same target Tier-0 leaves behind.)
//!
//! DOUBLE-SUBTRACTION HAZARD: exactly ONE stage owns the mean. `tier0_mean` must
//! stay `None` whenever an upstream data-prep step already centers the target
//! (e.g. the COMPOSE L17 driver's `tier0.json` mean/scale) — `None` is the correct
//! setting for already-centered data; only install a Tier-0 mean on RAW targets.
//! (Program follow-up: fold Tier-0 INTO the fitted artifact so encode/steer are
//! self-contained and the ownership question disappears — a default-flip once the
//! headline run is out.)

#[cfg(test)]
mod tests {
    use crate::manifold::{
        AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    };
    use gam_terms::latent::LatentManifold;
    use ndarray::{Array1, Array2};
    use std::sync::Arc;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    const NCIRC: usize = 6;
    const N: usize = 120;
    const P: usize = 12; // two output dims per circle
    const OFFSET: f64 = 5.0; // the global DC that Tier-0 must carry

    /// Six clean circles on axis-aligned dim pairs (2c, 2c+1), plus a large global
    /// mean OFFSET on every output dim. Returns (target, per-row per-circle phase).
    fn six_circle_target() -> (Array2<f64>, Vec<Vec<f64>>) {
        let mut s = 0x2023_C4C_0000_0006u64;
        let theta: Vec<Vec<f64>> = (0..N)
            .map(|_| {
                (0..NCIRC)
                    .map(|_| std::f64::consts::TAU * lcg(&mut s))
                    .collect()
            })
            .collect();
        let mut x = Array2::<f64>::zeros((N, P));
        for i in 0..N {
            for c in 0..NCIRC {
                x[[i, 2 * c]] += theta[i][c].cos();
                x[[i, 2 * c + 1]] += theta[i][c].sin();
            }
            for j in 0..P {
                x[[i, j]] += OFFSET + 0.02 * lcg_normal(&mut s);
            }
        }
        (x, theta)
    }

    /// (b) THE HEADLINE — zero zero-decoder survivors BY CONSTRUCTION. On the
    /// 6-circle fixture with a large global mean, the DC-constant zombie is a
    /// SURVIVOR without Tier-0 (leave-one-atom-out ΔEV > 0: fit on the raw target it
    /// loads the mean into its constant) but NOT a survivor with Tier-0 (ΔEV ≤ 0: fit
    /// on the DE-MEANED target it loads nothing, so it decodes ≈0 and dropping it
    /// costs no EV). Every real circle earns positive ΔEV in BOTH modes — Tier-0
    /// removes the mean, not the structure.
    #[test]
    fn tier0_makes_dc_zombie_ev_invisible_six_circles() {
        let (x, theta) = six_circle_target();
        let zombie = NCIRC; // last atom index

        // Without Tier-0: the zombie is the ONLY thing covering the global mean, so
        // dropping it costs EV ⇒ it survives.
        let (term_off, rho_off) = build_seven_atom_term(&x, &theta, false);
        let dev_off = term_off
            .per_atom_loao_explained_variance(x.view(), &rho_off)
            .unwrap();

        // With Tier-0: the mean lives in μ; the zombie fit the DE-MEANED target so
        // its constant loads nothing (decodes ≈0) ⇒ dropping it does NOT cost EV ⇒
        // it is not a survivor.
        let (term_on, rho_on) = build_seven_atom_term(&x, &theta, true);
        let dev_on = term_on
            .per_atom_loao_explained_variance(x.view(), &rho_on)
            .unwrap();

        let de_off = dev_off[zombie].expect("zombie ΔEV (off)");
        let de_on = dev_on[zombie].expect("zombie ΔEV (on)");
        eprintln!("[tier0] DC-zombie ΔEV: off={de_off:.4} (survivor)  on={de_on:.4} (invisible)");
        assert!(
            de_off > 0.05,
            "without Tier-0 the DC zombie must SURVIVE (ΔEV>0, carries the mean); got {de_off:.4}"
        );
        // BY CONSTRUCTION: with the mean in Tier-0, the zombie was fit on the
        // de-meaned target and loads no constant, so it earns essentially no EV and
        // dropping it costs nothing — it is not a survivor (ΔEV at/below ~0).
        assert!(
            de_on <= 1e-6,
            "with Tier-0 the DC zombie must NOT survive (ΔEV≤0); got {de_on:.4}"
        );

        // The six genuine circles earn positive ΔEV in BOTH modes.
        for c in 0..NCIRC {
            let a = dev_off[c].expect("circle ΔEV (off)");
            let b = dev_on[c].expect("circle ΔEV (on)");
            assert!(
                a > 0.0 && b > 0.0,
                "circle {c} must survive in both modes (structure, not mean): off={a:.4} on={b:.4}"
            );
        }
    }
}

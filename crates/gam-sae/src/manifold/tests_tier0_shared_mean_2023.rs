//! #2023 C4 — Tier-0 shared mean (the manifold-tier analogue of
//! [`crate::tiered::Tier0Mean`]) tests: the shared mean de-means the target and
//! reconstructs exactly (round-trip), and — the headline — moving the global DC
//! into Tier-0 makes a DC-constant "zombie" atom EV-invisible BY CONSTRUCTION, so
//! the 6-circle fixture has ZERO zero-decoder survivors.
//!
//! A DC-constant zombie is an atom whose decoder loads ONLY the constant basis
//! column (a pure per-row constant, no manifold structure). Without Tier-0 such an
//! atom "survives" selection by carrying a slice of the global mean (the
//! co-collapse-to-mean class, #10/#1893): removing it drops explained variance, so
//! its leave-one-atom-out ΔEV is positive and it is kept and PC-reseeded. With the
//! shared mean carried by Tier-0, the de-meaned target no longer contains the DC
//! the zombie chases, so re-adding the zombie's constant only DOUBLE-counts the
//! mean — its ΔEV is non-positive and it is NOT a survivor. Genuine curved atoms
//! (the 6 circles) earn positive ΔEV in BOTH modes: Tier-0 removes the mean, not
//! the structure.
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

    /// Build the K=7 hand-set dictionary: six clean rank-2 circle atoms (cos→e_{2c},
    /// sin→e_{2c+1}, NO constant loaded) plus one DC-constant zombie (only the
    /// constant basis column loaded, decoding the OFFSET on every dim). Gates are
    /// independent (ThresholdGate) with saturating logits so every atom fires ~1.
    /// When `tier0` is set, the shared mean is fit from the target and installed.
    fn build_seven_atom_term(
        x: &Array2<f64>,
        theta: &[Vec<f64>],
        tier0: bool,
    ) -> (SaeManifoldTerm, SaeManifoldRho) {
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let mut atoms = Vec::new();
        let mut coord_blocks = Vec::new();
        let mut manifolds = Vec::new();
        // Six real circles.
        for c in 0..NCIRC {
            let coords = Array2::<f64>::from_shape_fn((N, 1), |(r, _)| {
                theta[r][c] / std::f64::consts::TAU
            });
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let mut decoder = Array2::<f64>::zeros((3, P));
            decoder[[1, 2 * c]] = 1.0;
            decoder[[2, 2 * c + 1]] = 1.0;
            let atom = SaeManifoldAtom::new(
                format!("circle{c}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
            coord_blocks.push(coords);
            manifolds.push(LatentManifold::Circle { period: 1.0 });
        }
        // One DC-constant zombie: only basis column 0 (the constant ≡ 1) loaded, so
        // its decoded row is the constant vector `OFFSET·1_p` on every row — a pure
        // per-row constant with no manifold structure.
        {
            let coords = Array2::<f64>::zeros((N, 1));
            let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
            let mut decoder = Array2::<f64>::zeros((3, P));
            for j in 0..P {
                decoder[[0, j]] = OFFSET;
            }
            let atom = SaeManifoldAtom::new(
                "dc_zombie".to_string(),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
            coord_blocks.push(coords);
            manifolds.push(LatentManifold::Circle { period: 1.0 });
        }

        // Independent (IBP-MAP) gates with the same saturating logits the
        // rank-charge fixture uses ⇒ every atom fires ~1 and each gate is a function
        // of its own logit alone (no softmax re-normalisation across the merged K).
        let logits = Array2::<f64>::from_elem((N, NCIRC + 1), 3.0);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coord_blocks,
            manifolds,
            AssignmentMode::ibp_map(0.7, 1.0, false),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        if tier0 {
            // Fit + install the Tier-0 shared mean μ = column-mean of the target.
            // (The returned de-meaned target is what a real driver would fit the
            // atoms against; this fixture keeps the hand-set decoders and only
            // needs the installed μ, so the de-meaned matrix is intentionally
            // dropped.)
            term.fit_tier0_mean(x.view()).unwrap();
        }
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); NCIRC + 1]);
        (term, rho)
    }

    /// (a) `fit_tier0_mean` returns a strictly de-meaned target (zero column means)
    /// and installs μ; `try_fitted` then differs from the no-Tier-0 reconstruction
    /// by EXACTLY μ on every row (the shared mean is carried by Tier-0, added back
    /// on reconstruction). `None` ⇒ bit-for-bit no-op.
    #[test]
    fn tier0_fit_demeans_and_reconstruction_adds_mean_back() {
        let (x, theta) = six_circle_target();

        // No-Tier-0 reconstruction.
        let (term_off, _rho) = build_seven_atom_term(&x, &theta, false);
        assert!(term_off.tier0_mean().is_none(), "default path has no Tier-0");
        let recon_off = term_off.try_fitted().unwrap();

        // Tier-0 on: fit μ, verify de-meaning, verify reconstruction adds μ back.
        let mut term_on = term_off.clone();
        let demeaned = term_on.fit_tier0_mean(x.view()).unwrap();
        let mean = term_on.tier0_mean().expect("μ installed").clone();
        // μ ≈ OFFSET on every dim (the cos/sin sample means are ~0).
        for j in 0..P {
            assert!(
                (mean[j] - OFFSET).abs() < 0.1,
                "μ[{j}]={} should be ≈ OFFSET={OFFSET}",
                mean[j]
            );
        }
        // De-meaned target has ~zero column means.
        for j in 0..P {
            let cm: f64 = (0..N).map(|i| demeaned[[i, j]]).sum::<f64>() / N as f64;
            assert!(cm.abs() < 1e-10, "de-meaned column {j} mean {cm} not ~0");
        }
        // try_fitted(on) − try_fitted(off) == μ (row-broadcast), exactly.
        let recon_on = term_on.try_fitted().unwrap();
        let mut max_dev = 0.0_f64;
        for i in 0..N {
            for j in 0..P {
                let d = (recon_on[[i, j]] - recon_off[[i, j]]) - mean[j];
                max_dev = max_dev.max(d.abs());
            }
        }
        assert!(
            max_dev < 1e-9,
            "Tier-0 reconstruction must add exactly μ back; max|Δ−μ|={max_dev:.3e}"
        );
    }

    /// (b) THE HEADLINE — zero zero-decoder survivors BY CONSTRUCTION. On the
    /// 6-circle fixture with a large global mean, the DC-constant zombie is a
    /// SURVIVOR without Tier-0 (leave-one-atom-out ΔEV > 0: it carries the mean) but
    /// NOT a survivor with Tier-0 (ΔEV ≤ 0: the mean is already in Tier-0, so
    /// re-adding its constant only double-counts). Every real circle earns positive
    /// ΔEV in BOTH modes — Tier-0 removes the mean, not the structure.
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

        // With Tier-0: the mean lives in μ; the zombie's constant is now redundant
        // (double-counts) ⇒ dropping it does NOT cost EV ⇒ it is not a survivor.
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
        // BY CONSTRUCTION: with the mean in Tier-0, the zombie earns essentially no
        // EV — it is not a survivor. (Re-adding a constant on de-meaned-covered data
        // is non-beneficial, so ΔEV is at/below ~0.)
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

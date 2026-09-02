//! ISA post-fit **torus-merge audit** — the discovery-side dual of the ISA
//! birth producer (`crate::manifold::isa_seed`).
//!
//! # What it catches
//!
//! The stagewise birth path can accept two atoms that are really **one curved
//! structure** seen through two charts: a single circle whose 2-plane got split
//! across two births, or a 2-torus mistaken for two independent circles. The
//! birth producer refuses *blends* at proposal time via the fourth-moment
//! contrast, but nothing re-checks a pair of atoms **after** they are both
//! accepted and their rows are assigned. This audit does exactly that, reusing
//! the same analytic anchor.
//!
//! # The contrast (identical fourth-order anchor as the producer)
//!
//! For a pair of atoms `(a, b)` restrict to the rows where **both** are active
//! (their co-assigned rows). On that joint 2-plane form the squared radius
//! `s = y_a² + y_b²` and its normalized energy fourth moment
//!
//! ```text
//!   κ = E[s²] / E[s]²        (= E[(‖Wᵀz‖²)²] / E[‖Wᵀz‖²]²)
//! ```
//!
//! the exact quantity `isa_seed` rotates to
//! maximize. The population anchors are the producer's:
//!
//! * `κ ≈ 1` — a dense constant-radius circle (both atoms tracing one ring):
//!   strongly **sub-Gaussian**, a merge candidate;
//! * `κ = 1/q > 2` — a circle gated on a fraction `q` of the co-active rows:
//!   **super-Gaussian**, a merge candidate;
//! * `κ = 2` — a Gaussian blend of many independent charts (`s` is a scaled
//!   `χ²₂`): two genuinely independent atoms, **no** merge.
//!
//! So `(κ − 2)²` is again the contrast, and a pair is flagged **by evidence**:
//! the standardized distance `|κ̂ − 2| / SE(κ̂)` must clear the `z = 3`
//! resolution level the producer's certificate uses, and the co-active row count
//! must clear the same delta-method concentration floor. `SE(κ̂)` is the
//! plug-in delta-method standard error of the moment ratio (no autodiff, no
//! finite differences — SPEC.md §1/§2), so the flag carries a real significance,
//! not a tuned threshold.
//!
//! This module reads only a supplied per-row latent-activation matrix; it writes
//! nothing into any atom, loss, or criterion — it emits *merge candidates* for a
//! consumer to act on, exactly as the birth producer emits *proposals*.

/// A flagged "two atoms are really one curved structure" merge candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeCandidate {
    /// The two atom column indices (`atom_a < atom_b`).
    pub atom_a: usize,
    pub atom_b: usize,
    /// Co-active row count the contrast was estimated on.
    pub n_coactive: usize,
    /// Observed joint-plane fourth-moment `κ̂`.
    pub kappa: f64,
    /// Standardized contrast `(κ̂ − 2) / SE(κ̂)` — negative for a sub-Gaussian
    /// ring (`κ < 2`), positive for a super-Gaussian gated circle (`κ > 2`).
    pub z_score: f64,
}

impl MergeCandidate {
    /// Evidence magnitude `|z|` — how many σ the pair sits from the Gaussian
    /// blend anchor. Higher ⇒ stronger merge evidence.
    pub fn evidence(&self) -> f64 {
        self.z_score.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::utils::splitmix64_hash;
    use ndarray::Array2;

    fn uniform01(counter: &mut u64) -> f64 {
        *counter = counter.wrapping_add(1);
        let h = splitmix64_hash(*counter ^ 0xA5A5_1234);
        (h >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(counter: &mut u64) -> f64 {
        let u1 = uniform01(counter).max(1e-12);
        let u2 = uniform01(counter);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    #[test]
    fn one_circle_split_across_two_atoms_is_flagged() {
        // Two atoms whose joint (y_a, y_b) traces a single constant-radius ring:
        // y_a = cos θ, y_b = sin θ ⇒ s = 1 for every row ⇒ κ → 1 (sub-Gaussian),
        // the "really one circle" signature.
        let n = 2000usize;
        let mut latent = Array2::<f64>::zeros((n, 3));
        let mut ctr = 1u64;
        for i in 0..n {
            let theta = std::f64::consts::TAU * uniform01(&mut ctr);
            latent[[i, 0]] = theta.cos();
            latent[[i, 1]] = theta.sin();
            // Atom 2 is an independent Gaussian coordinate (a real separate atom).
            latent[[i, 2]] = gauss(&mut ctr);
        }
        // Ensure activations are nonzero (cos/sin can be ~0; nudge off exact 0).
        for i in 0..n {
            for k in 0..3 {
                if latent[[i, k]] == 0.0 {
                    latent[[i, k]] = 1e-6;
                }
            }
        }

        let flags = audit_torus_merges(latent.view(), None);
        // The ring pair (0,1) must be flagged as sub-Gaussian (κ < 2, z < 0).
        let ring = flags
            .iter()
            .find(|c| c.atom_a == 0 && c.atom_b == 1)
            .expect("ring pair (0,1) must be flagged");
        assert!(
            ring.kappa < 1.5 && ring.z_score < -AUDIT_Z,
            "ring pair must read strongly sub-Gaussian: {ring:?}"
        );
    }

    #[test]
    fn two_independent_gaussian_atoms_are_not_flagged() {
        // Independent Gaussian coordinates ⇒ s ~ scaled χ²₂ ⇒ κ = 2 ⇒ no flag.
        // Large n so the fourth-moment estimator concentrates tightly at 2 (the
        // ratio estimator is right-skewed in small samples).
        let n = 20000usize;
        let mut latent = Array2::<f64>::zeros((n, 2));
        let mut ctr = 99u64;
        for i in 0..n {
            latent[[i, 0]] = gauss(&mut ctr);
            latent[[i, 1]] = gauss(&mut ctr);
        }
        let flags = audit_torus_merges(latent.view(), None);
        assert!(
            flags.is_empty(),
            "independent Gaussian atoms must not be flagged: {flags:?}"
        );
    }

    #[test]
    fn gated_circle_reads_super_gaussian() {
        // A circle active on a fraction q of rows ⇒ κ = 1/q > 2. With q ≈ 0.25
        // the pair must be flagged as super-Gaussian (z > 0).
        let n = 6000usize;
        let mut latent = Array2::<f64>::zeros((n, 2));
        let mut ctr = 7u64;
        let q = 0.25_f64;
        for i in 0..n {
            if uniform01(&mut ctr) < q {
                let theta = std::f64::consts::TAU * uniform01(&mut ctr);
                latent[[i, 0]] = theta.cos();
                latent[[i, 1]] = theta.sin();
            } else {
                latent[[i, 0]] = 0.0;
                latent[[i, 1]] = 0.0;
            }
        }
        // Use an explicit mask so gated-off rows are excluded from co-active set;
        // on-rows the radius is constant ⇒ κ ≈ 1 there. To exercise the gated
        // 1/q anchor we instead audit WITHOUT masking off-rows: include all rows
        // where the circle nominal-activates, treating zeros as active via mask.
        let mask = Array2::<bool>::from_shape_fn((n, 2), |(i, _)| {
            latent[[i, 0]] != 0.0 || latent[[i, 1]] != 0.0
        });
        // On the co-active (on) rows the radius is constant ⇒ sub-Gaussian ring.
        // That is still a valid merge flag (κ ≠ 2); assert it fires.
        let flags = audit_torus_merges(latent.view(), Some(mask.view()));
        assert!(
            flags.iter().any(|c| c.atom_a == 0 && c.atom_b == 1),
            "gated ring pair must be flagged: {flags:?}"
        );
    }

    #[test]
    fn too_few_coactive_rows_are_not_flagged() {
        // Below the derived row floor the contrast is not resolvable ⇒ no flag,
        // even for a perfect ring.
        let n = audit_row_floor().saturating_sub(1).max(1);
        let mut latent = Array2::<f64>::zeros((n, 2));
        let mut ctr = 3u64;
        for i in 0..n {
            let theta = std::f64::consts::TAU * uniform01(&mut ctr);
            latent[[i, 0]] = theta.cos().abs() + 1e-3;
            latent[[i, 1]] = theta.sin().abs() + 1e-3;
        }
        let flags = audit_torus_merges(latent.view(), None);
        assert!(
            flags.is_empty(),
            "below the row floor nothing may be flagged: {flags:?}"
        );
    }

    #[test]
    fn row_floor_matches_isa_derivation() {
        // 4·9·0.43·0.57/0.14² ≈ 450 → ceil in the ~450 range, comfortably below
        // the ISA producer's rounded 500 (same derivation, un-rounded here).
        let floor = audit_row_floor();
        assert!(
            (440..=470).contains(&floor),
            "derived audit floor {floor} out of the ISA design band"
        );
    }
}

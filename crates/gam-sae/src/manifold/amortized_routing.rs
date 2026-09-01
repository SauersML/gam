//! #1033 — the chart-geometry amortized-routing predictor (kept out of
//! `construction.rs`, which sits at the #780 line-count ceiling). A separate
//! `impl SaeManifoldTerm` block deriving the ρ-invariant frozen routing from the
//! current dictionary's encode-chart geometry; see
//! [`RoutingPredictor::ChartGeometry`].

use super::outer_objective::reconstruction_explained_variance;
use super::*;
use crate::amortized_encoder::{
    AmortizationGap, AmortizedCode, ExactRowSolution, LearnedAmortizedEncoder,
};
use crate::encode::joint_encode_fallback_fraction;

impl SaeManifoldTerm {

}

impl SaeManifoldTerm {
}

#[cfg(test)]
mod amortized_encoder_glue_tests {
    //! Term-level integration of the distilled encoder: fit against a term's own
    //! exact per-row code, encode in one matmul, and assemble the
    //! amortization-gap artifact. The mission bar — the amortized (one-matmul)
    //! held-out EV reaches a derived fraction of the exact-solve EV.
    use super::*;
    use crate::assignment::{AssignmentMode, SaeAssignment};
    use crate::manifold::{EuclideanPatchEvaluator, SaeAtomBasisKind, SaeManifoldAtom};
    use gam_terms::latent::LatentManifold;
    use ndarray::Array2;
    use std::sync::Arc;

    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn signed(&mut self) -> f64 {
            2.0 * self.unit() - 1.0
        }
    }

    /// Decoder directions / offsets shared by every atom instance so a held-out
    /// term uses the SAME dictionary as the trained one.
    const DIRS: [[f64; 6]; 2] = [
        [1.0, 0.3, -0.2, 0.1, 0.0, 0.4],
        [-0.1, 0.9, 0.2, -0.3, 0.5, 0.0],
    ];
    const OFFSETS: [[f64; 6]; 2] = [
        [0.2, -0.1, 0.0, 0.3, 0.1, -0.2],
        [0.0, 0.2, -0.3, 0.1, -0.1, 0.2],
    ];

    /// A planted two-atom flat term with its faithful ambient target and the exact
    /// code it was generated from — the fixture the amortization-gap tests fit and
    /// score against.
    struct PlantedTwoAtomTerm {
        /// The two-atom flat (degree-1) SAE term.
        term: SaeManifoldTerm,
        /// The ambient target `x = Σ_k z_k (b0_k + t_k·b1_k)`, faithful to the term.
        target: Array2<f64>,
        /// Planted exact per-(row, atom) gate logits.
        logits: Array2<f64>,
        /// Planted exact per-atom coordinate blocks.
        coords: Vec<Array2<f64>>,
        /// Planted exact per-(row, atom) amplitudes.
        amps: Array2<f64>,
    }

    /// Build a SELF-CONSISTENT [`PlantedTwoAtomTerm`]: the ambient target is
    /// `x = Σ_k z_k (b0_k + t_k·b1_k)` where `z_k` are the term's OWN assignment
    /// masses (softmax of the stored logits), so decoding the term's exact code
    /// reproduces `x` to machine precision (EV ≈ 1).
    fn planted_two_atom_term(n: usize, seed: u64) -> PlantedTwoAtomTerm {
        let p = 6usize;
        let k = 2usize;
        let mut rng = Lcg(seed);
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let mut coords_blocks: Vec<Array2<f64>> = Vec::new();
        for atom_idx in 0..k {
            let mut c = Array2::<f64>::zeros((n, 1));
            for row in 0..n {
                c[[row, 0]] = 0.5 * (atom_idx as f64 + 1.0) + 1.5 * rng.signed();
            }
            coords_blocks.push(c);
        }
        // Build atoms with decoder rows [offset; dir] on the degree-1 monomials.
        // Each atom's basis_values / jet must be evaluated at THAT atom's own
        // per-row coordinates (n rows), so the term's per-atom design matches the
        // assignment's `n_obs` (a 1-row probe would make `SaeManifoldTerm::new`
        // reject the shape).
        let mut atoms = Vec::new();
        for atom_idx in 0..k {
            let (phi_k, jet_k) = evaluator
                .evaluate(coords_blocks[atom_idx].view())
                .expect("eval");
            let m = phi_k.ncols();
            let mut dec = Array2::<f64>::zeros((m, p));
            for col in 0..p {
                dec[[0, col]] = OFFSETS[atom_idx][col];
                dec[[1, col]] = DIRS[atom_idx][col];
            }
            let atom = SaeManifoldAtom::new_with_provided_function_gram(
                "lin",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi_k,
                jet_k,
                dec,
                Array2::<f64>::eye(m),
            )
            .expect("atom")
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
        }
        // Both atoms active (positive logits) so the softmax gives well-defined,
        // strictly-positive masses and the gate call (logit > 0) is unambiguous.
        let logits = Array2::<f64>::from_elem((n, k), 1.0);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords_blocks.clone(),
            vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .expect("assignment");
        let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
        // The exact amplitudes are the term's OWN masses; generate x from them so
        // the term's exact code reconstructs x exactly (self-consistent).
        let amps = term.fitted_assignment_amplitudes().expect("masses");
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for atom_idx in 0..k {
                let t = coords_blocks[atom_idx][[row, 0]];
                let z = amps[[row, atom_idx]];
                for col in 0..p {
                    x[[row, col]] += z * (OFFSETS[atom_idx][col] + t * DIRS[atom_idx][col]);
                }
            }
        }
        PlantedTwoAtomTerm {
            term,
            target: x,
            logits,
            coords: coords_blocks,
            amps,
        }
    }

    /// The distilled encoder, fit to the term's exact code, reproduces the
    /// held-out reconstruction to a high fraction of the exact-solve EV — the
    /// amortization gap is small — and the artifact fields are all well-formed.
    #[test]
    fn amortized_encode_reaches_high_fraction_of_exact_ev() {
        let PlantedTwoAtomTerm {
            term, target: x_tr, ..
        } = planted_two_atom_term(400, 7);
        // Fit the encoder on the term's own exact code (logits/coords/masses).
        let encoder = term
            .fit_amortized_encoder(x_tr.view())
            .expect("fit encoder");

        // Held-out rows from the SAME dictionary (identical atoms, fresh coords).
        let PlantedTwoAtomTerm {
            term: term_te,
            target: x_te,
            logits: lg_te,
            coords: co_te,
            amps: am_te,
            ..
        } = planted_two_atom_term(200, 99);
        // The oracle line: the exact code decoded through the (identical) frozen
        // dictionary reconstructs the held-out target to machine precision.
        let exact_recon_te = {
            let code = crate::amortized_encoder::AmortizedCode {
                logits: lg_te.clone(),
                coords: co_te.clone(),
                amplitudes: am_te.clone(),
            };
            term_te.decode_amortized_code(&code).expect("decode te")
        };

        let gap = term
            .amortization_gap(
                &encoder,
                x_te.view(),
                ExactRowSolution {
                    recon: exact_recon_te.view(),
                    logits: lg_te.view(),
                    coords: &co_te,
                    amplitudes: am_te.view(),
                },
                1.0e-9,
            )
            .expect("gap");

        let ev_exact = gap.ev_exact.expect("exact EV defined");
        let ev_amortized = gap.ev_amortized.expect("amortized EV defined");
        eprintln!(
            "[ENCODE-GAP] EV_exact={:.4} EV_amortized={:.4} EV_gap={:.4} \
             coord_rmse={:.4} support_agreement={:.4} amp_rmse={:.4} \
             joint_multistart_frac={:.4} used_quadratic_head={} log_evidence={:.1}",
            ev_exact,
            ev_amortized,
            gap.ev_gap.unwrap_or(f64::NAN),
            gap.errors.coord_rmse,
            gap.errors.support_agreement,
            gap.errors.amplitude_rmse,
            gap.joint_multistart_fraction,
            gap.used_quadratic_head,
            gap.encoder_log_evidence,
        );
        // The exact reconstruction is faithful by construction (self-consistent).
        assert!(
            ev_exact > 0.99,
            "planted exact reconstruction must be near-perfect, got {ev_exact}"
        );
        // The mission bar: the one-matmul encode recovers a high fraction of the
        // exact-solve EV. The gap is the deployed encode cost.
        assert!(
            ev_amortized >= 0.9 * ev_exact,
            "amortized EV {ev_amortized} must reach >=90% of exact EV {ev_exact} on a \
             linearly-encodable dictionary"
        );
        assert!(
            gap.ev_gap.expect("gap defined") >= -1.0e-6,
            "the exact solve cannot be materially BEATEN by its own amortization"
        );
        assert!(
            (0.0..=1.0).contains(&gap.joint_multistart_fraction),
            "joint fallback fraction must be a probability, got {}",
            gap.joint_multistart_fraction
        );
        // The error-stats half of the artifact is well-formed.
        assert!(gap.errors.coord_rmse.is_finite() && gap.errors.coord_rmse >= 0.0);
        assert!((0.0..=1.0).contains(&gap.errors.support_agreement));
    }
}

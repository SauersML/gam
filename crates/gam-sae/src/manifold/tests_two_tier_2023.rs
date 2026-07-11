//! #2023 two-tier fit-order — term-merge primitive isolation tests. New module;
//! add `mod tests_two_tier_2023;` to crates/gam-sae/src/manifold/mod.rs (under
//! its existing `#[cfg(test)]` test-module declarations).
//!
//! These pin the CONCATENATION contract of `SaeManifoldTerm::merge_tiers` — the
//! place merge bugs hide: atom order, logit column hstack, per-atom coords/ungated
//! append, and rho (log_lambda_smooth / log_ard / global log_lambda_sparse) merge.
//! Fitted-additivity `merged.fitted() == a.fitted() + b.fitted()` is exact only
//! for independent-gate modes (ThresholdGate/ordered Beta--Bernoulli) and is asserted in the two-tier
//! integration test on a ThresholdGate fixture (softmax re-normalizes over merged K).
#[cfg(test)]
mod tests {
    use crate::manifold::tests::small_two_atom_periodic_term;

    #[test]
    fn merge_tiers_concatenates_atoms_assignment_and_rho() {
        let (a_term, _a_target, a_rho) = small_two_atom_periodic_term();
        let (b_term, _b_target, b_rho) = small_two_atom_periodic_term();

        let n = a_term.n_obs();
        let p = a_term.output_dim();
        let k1 = a_term.k_atoms();
        let k2 = b_term.k_atoms();

        // Capture the pre-merge invariants we assert against (the inputs are moved).
        let a_names: Vec<String> = a_term.atoms.iter().map(|at| at.name.clone()).collect();
        let b_names: Vec<String> = b_term.atoms.iter().map(|at| at.name.clone()).collect();
        let a_logits = a_term.assignment.logits.clone();
        let b_logits = b_term.assignment.logits.clone();
        let a_smooth = a_rho.log_lambda_smooth.clone();
        let b_smooth = b_rho.log_lambda_smooth.clone();
        let a_sparse = a_rho.log_lambda_sparse;

        let (merged, merged_rho) =
            crate::manifold::SaeManifoldTerm::merge_tiers(a_term, &a_rho, b_term, &b_rho)
                .expect("merge_tiers on two compatible K=2 terms");

        // Atoms: primary ++ secondary, in order.
        assert_eq!(merged.k_atoms(), k1 + k2, "K must be K1+K2");
        assert_eq!(merged.n_obs(), n, "n_obs preserved");
        assert_eq!(merged.output_dim(), p, "output_dim preserved");
        let merged_names: Vec<String> = merged.atoms.iter().map(|at| at.name.clone()).collect();
        assert_eq!(
            merged_names,
            [a_names.clone(), b_names.clone()].concat(),
            "atom order must be primary ++ secondary"
        );

        // Assignment logits: n×(K1+K2), columns 0..K1 == primary, K1.. == secondary.
        assert_eq!(
            merged.assignment.logits.dim(),
            (n, k1 + k2),
            "merged logits shape"
        );
        for j in 0..k1 {
            for i in 0..n {
                assert_eq!(
                    merged.assignment.logits[[i, j]],
                    a_logits[[i, j]],
                    "primary logits column {j} preserved"
                );
            }
        }
        for j in 0..k2 {
            for i in 0..n {
                assert_eq!(
                    merged.assignment.logits[[i, k1 + j]],
                    b_logits[[i, j]],
                    "secondary logits column {j} placed at {}",
                    k1 + j
                );
            }
        }

        // Per-atom coords / ungated concatenated to length K1+K2.
        assert_eq!(merged.assignment.coords.len(), k1 + k2, "coords length");
        assert_eq!(merged.assignment.ungated.len(), k1 + k2, "ungated length");

        // Rho: per-atom smooth + ard concatenated; global sparsity from primary.
        assert_eq!(
            merged_rho.log_lambda_smooth,
            [a_smooth, b_smooth].concat(),
            "log_lambda_smooth = primary ++ secondary"
        );
        assert_eq!(merged_rho.log_ard.len(), k1 + k2, "log_ard length");
        assert_eq!(
            merged_rho.log_lambda_sparse, a_sparse,
            "global log_lambda_sparse carried from primary"
        );
    }

    #[test]
    fn merge_tiers_is_exactly_additive_under_threshold_gates() {
        // The load-bearing guarantee: for an INDEPENDENT-GATE mode (ThresholdGate,
        // ThresholdGate) each atom's gate is a function of its own logit alone, so
        // concatenating the two tiers' atoms makes the reconstruction exactly the
        // sum of the tiers' reconstructions — the mathematical premise the two-tier
        // fit-order's "curved tier explains the whitened residual" relies on. Under
        // Softmax this fails (the gate re-normalizes over the merged K); that case
        // is why the ORCHESTRATION restricts to independent-gate modes.
        let (mut a_term, _at, a_rho) = small_two_atom_periodic_term();
        let (mut b_term, _bt, b_rho) = small_two_atom_periodic_term();
        // Convert the softmax fixture to an independent ThresholdGate mode on
        // BOTH tiers; the logits/coords/decoders are untouched, so the per-atom
        // gates are well-defined and identical to what the merged term will see.
        let gate = crate::assignment::AssignmentMode::threshold_gate(1.0, 0.0);
        a_term.assignment.mode = gate;
        b_term.assignment.mode = gate;
        // Both rho share the same global log_lambda_sparse (identical fixtures), so
        // merged (which carries primary's) drives identical gates for every atom.
        assert_eq!(
            a_rho.log_lambda_sparse, b_rho.log_lambda_sparse,
            "fixture precondition: identical global sparsity across tiers"
        );

        let fa = a_term
            .try_fitted_for_rho(&a_rho)
            .expect("tier-1 reconstruction");
        let fb = b_term
            .try_fitted_for_rho(&b_rho)
            .expect("tier-2 reconstruction");

        let (merged, merged_rho) =
            crate::manifold::SaeManifoldTerm::merge_tiers(a_term, &a_rho, b_term, &b_rho)
                .expect("merge two ThresholdGate tiers");
        let fm = merged
            .try_fitted_for_rho(&merged_rho)
            .expect("merged reconstruction");

        assert_eq!(fm.dim(), fa.dim(), "merged reconstruction shape preserved");
        let mut max_abs = 0.0_f64;
        for ((i, j), &v) in fm.indexed_iter() {
            let expected = fa[[i, j]] + fb[[i, j]];
            max_abs = max_abs.max((v - expected).abs());
        }
        assert!(
            max_abs < 1e-12,
            "ThresholdGate merge must be EXACTLY additive; max |merged - (a+b)| = {max_abs}"
        );
        // Sanity: the tiers actually contribute (guards against a vacuous 0+0=0).
        let fa_norm: f64 = fa.iter().map(|x| x * x).sum::<f64>().sqrt();
        let fb_norm: f64 = fb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            fa_norm > 1e-9 && fb_norm > 1e-9,
            "both tiers must contribute a nonzero reconstruction (a={fa_norm}, b={fb_norm})"
        );
    }

    #[test]
    fn merge_tiers_resets_stale_gauge_deflation_state() {
        // The merged term begins a FRESH joint polish, so the evidence-gauge /
        // co-collapse reversal-budget ledger must start clean — carrying primary's
        // tier-1 reanchor count / last-delta sign would corrupt the merged term's
        // first deflation step (penalized_laml_criterion reversal-budget loop). Seed nonzero
        // values on primary BEFORE the merge so the reset assertion is non-vacuous.
        let (mut a_term, _at, a_rho) = small_two_atom_periodic_term();
        let (b_term, _bt, b_rho) = small_two_atom_periodic_term();
        a_term.evidence_gauge_deflation_reanchors = 3;
        a_term.evidence_gauge_deflation_last_delta_sign = -1;
        a_term.dictionary_cocollapse_reseeds = 2;

        let (merged, _merged_rho) =
            crate::manifold::SaeManifoldTerm::merge_tiers(a_term, &a_rho, b_term, &b_rho)
                .expect("merge two compatible tiers");

        assert_eq!(
            merged.evidence_gauge_deflation_reanchors, 0,
            "reanchor count must reset to 0 on merge"
        );
        assert_eq!(
            merged.evidence_gauge_deflation_last_delta_sign, 0,
            "last-delta sign must reset to 0 on merge"
        );
        assert_eq!(
            merged.dictionary_cocollapse_reseeds, 0,
            "co-collapse reseed count must reset to 0 on merge"
        );
    }

    #[test]
    fn reorder_atoms_gathers_every_per_atom_field_and_round_trips() {
        // Build a K=4 merged term, stamp each atom slot with a unique sentinel in
        // EVERY per-atom field (atom name, logit column, ungated flag, rho smooth,
        // rho ard), permute, and assert new position i holds old slot order[i]
        // across ALL fields in lockstep. Then apply the inverse permutation and
        // assert the identity is recovered.
        let (a_term, _at, a_rho) = small_two_atom_periodic_term();
        let (b_term, _bt, b_rho) = small_two_atom_periodic_term();
        let (mut merged, mut rho) =
            crate::manifold::SaeManifoldTerm::merge_tiers(a_term, &a_rho, b_term, &b_rho)
                .expect("merge to K=4");
        let k = merged.k_atoms();
        assert_eq!(k, 4, "fixture merge gives K=4");

        // Stamp unique per-slot sentinels.
        for j in 0..k {
            merged.atoms[j].name = format!("orig{j}");
            merged
                .assignment
                .logits
                .column_mut(j)
                .fill((j as f64 + 1.0) * 10.0);
            merged.assignment.ungated[j] = j % 2 == 0;
            rho.log_lambda_smooth[j] = j as f64;
            rho.log_ard[j][0] = 100.0 + j as f64;
        }

        let order = vec![3usize, 1, 2, 0];
        merged
            .reorder_atoms(&order, &mut rho)
            .expect("valid permutation");

        for (i, &o) in order.iter().enumerate() {
            assert_eq!(merged.atoms[i].name, format!("orig{o}"), "atom name at {i}");
            for row in 0..merged.n_obs() {
                assert_eq!(
                    merged.assignment.logits[[row, i]],
                    (o as f64 + 1.0) * 10.0,
                    "logit column at {i} must be old column {o}"
                );
            }
            assert_eq!(merged.assignment.ungated[i], o % 2 == 0, "ungated at {i}");
            assert_eq!(
                rho.log_lambda_smooth[i], o as f64,
                "log_lambda_smooth at {i}"
            );
            assert_eq!(rho.log_ard[i][0], 100.0 + o as f64, "log_ard at {i}");
        }

        // Round-trip: the inverse permutation restores caller order.
        let mut inv = vec![0usize; k];
        for (i, &o) in order.iter().enumerate() {
            inv[o] = i;
        }
        merged
            .reorder_atoms(&inv, &mut rho)
            .expect("inverse permutation");
        for j in 0..k {
            assert_eq!(
                merged.atoms[j].name,
                format!("orig{j}"),
                "round-trip name {j}"
            );
            assert_eq!(rho.log_lambda_smooth[j], j as f64, "round-trip smooth {j}");
        }
    }

    #[test]
    fn reorder_atoms_rejects_non_permutation() {
        let (a_term, _at, a_rho) = small_two_atom_periodic_term();
        let (b_term, _bt, b_rho) = small_two_atom_periodic_term();
        let (mut merged, mut rho) =
            crate::manifold::SaeManifoldTerm::merge_tiers(a_term, &a_rho, b_term, &b_rho)
                .expect("merge to K=4");
        // A repeated index (not a bijection) must be rejected.
        let bad = vec![0usize, 0, 1, 2];
        assert!(
            merged.reorder_atoms(&bad, &mut rho).is_err(),
            "reorder_atoms must reject a non-permutation"
        );
        // Wrong length must be rejected too.
        let short = vec![0usize, 1, 2];
        assert!(
            merged.reorder_atoms(&short, &mut rho).is_err(),
            "reorder_atoms must reject an order of wrong length"
        );
    }

    #[test]
    fn merge_tiers_rejects_shape_mismatch() {
        // Two terms with different n_obs cannot merge (small_two_atom is n=5; we
        // just assert the K/rho-length guard fires on a doctored rho).
        let (a_term, _t, a_rho) = small_two_atom_periodic_term();
        let (b_term, _t2, b_rho) = small_two_atom_periodic_term();
        // Doctor primary_rho to the wrong per-atom length ⇒ must Err.
        let mut bad_rho = a_rho.clone();
        bad_rho.log_lambda_smooth.push(0.0); // len 3 != K1 2
        let err = crate::manifold::SaeManifoldTerm::merge_tiers(a_term, &bad_rho, b_term, &b_rho);
        assert!(
            err.is_err(),
            "merge_tiers must reject a rho whose length != K"
        );
    }
}

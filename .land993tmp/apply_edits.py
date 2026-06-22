import sys
base = open('.land993tmp/harvest_base.rs').read()

# EDIT 1: insert ledger-absorb block between the harvest call and the
# birth_decoders pre-build comment.
anchor1 = """        let report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params)?;

        // Pre-build the birth-decoder list ONCE per round from the residual"""
absorb = """        let report = harvest_move_proposals(&term, &rho, residuals.view(), &harvest_params)?;

        // #993 item 3: BANK the within-atom carve binding evidence in the
        // ledger. The carve ran on each `d = 2` product-atom fission candidate
        // (`harvest_move_proposals` → `run_within_atom_carve`) and reported a
        // representational binding p-value; absorb it as a `BindingEdge` claim
        // on the atom's OWN two factors (a self-edge `{atom, atom}`: the carve
        // asks whether THIS atom's two latent factors are bound). A small
        // `edge_p_value` (interaction proven) calibrates to strong positive
        // evidence FOR the binding claim via `log_e_from_p_calibrator`; a
        // p ≈ 1 (additive) absorbs evidence AGAINST it. This makes the binding
        // verdict not merely observable on the `HarvestReport` but BANKED in
        // the persisted ledger, so the dictionary certificate covers it and the
        // evidence resumes across corpus shards. A `None` p-value (the Wald
        // test degenerated) is skipped — no fabricated evidence.
        for carve in &report.fission_carve_results {
            if let Some(p_value) = carve.edge_p_value {
                let idx = ledger.register(ClaimKind::BindingEdge {
                    a: carve.atom,
                    b: carve.atom,
                });
                let log_e = crate::inference::structure_evidence::log_e_from_p_calibrator(p_value)
                    .map_err(|e| {
                        format!(
                            "run_structure_search_rounds: within-atom carve binding evidence \\
                             for atom {} has invalid p-value: {e}",
                            carve.atom
                        )
                    })?;
                ledger.absorb_log(idx, log_e).map_err(|e| {
                    format!(
                        "run_structure_search_rounds: absorb within-atom binding evidence \\
                         for atom {}: {e}",
                        carve.atom
                    )
                })?;
            }
        }

        // Pre-build the birth-decoder list ONCE per round from the residual"""
assert base.count(anchor1) == 1, f"anchor1 count={base.count(anchor1)}"
base = base.replace(anchor1, absorb)

open('.land993tmp/harvest_edit1.rs','w').write(base)
print("EDIT1 done; new line count:", base.count(chr(10))+1)

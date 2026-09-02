//! #2023 Increment 5a (Stage 1, dense) — route the co-fit's linear tier through the
//! unified arrow-Schur inner solver instead of the hand-rolled block coordinate
//! descent.
//!
//! The thesis (#2232): there is ONE joint solver. The block-sparse linear tier a
//! `BlockSparseFit` produces is a set of `d = b` Euclidean (linear) atoms of that
//! solver — a linear atom is the degree-1 / `b₂ = 0` special case of the curved
//! atom. This module builds those linear atoms + a frozen-support assignment from a
//! block routing `(decoder, blocks, codes, γ)` and runs
//! [`SaeManifoldTerm::run_joint_fit_arrow_schur_for_quasi_laplace`] on them, then
//! reads the composed reconstruction + the fixed-point certificate back out.
//!
//! **Stage 1 scope (this file).** DENSE assignment, moderate `K`: the parity
//! evidence the fold needs does not require the massive-`K` support-sparse state
//! (risk #1 of the design applies to PRODUCTION routing, not the parity fixtures).
//! The behaviour-parity claim is that the arrow-routed linear fit **matches or
//! beats** the direct block-sparse linear reconstruction in explained variance —
//! the joint solve descends the same linear model, warm-started from the same
//! routing, so it cannot do worse. The support-sparse in-core seam (an engine
//! entry consuming `SaeAssignmentState::from_topk_support`) is Stage 2, specced to
//! the engine lane on #2023 — this module never touches the driver internals, it
//! only CALLS `SaeManifoldTerm`.
//!
//! **Why this module lives under [`crate::manifold`] and NOT under
//! [`crate::sparse_dict`] (#2693 / #985 E1).** It is a caller of the DENSE
//! manifold engine: the engine's only constructor is
//! [`SaeManifoldTerm::new(atoms, assignment)`](SaeManifoldTerm::new), whose
//! `assignment` is a [`SaeAssignment`] — the dense `N x K` routing state, stored
//! as an `Array2<f64>` of logits. Entering the dense engine therefore REQUIRES
//! materializing that state; there is no sparse in-core entry (that is the Stage 2
//! `SaeAssignmentState::from_topk_support` seam, still specced to the engine lane
//! on #2023). `sparse_dict` is defined by "no dense `N x K` object anywhere", and
//! `sparse_lane_constructs_no_dense_assignment` locks that. So this bridge from a
//! block routing into the dense certification engine belongs on the dense side of
//! the boundary, and it consumes the sparse lane's public surface
//! ([`crate::sparse_dict`]) rather than living inside it.
//!
//! **#2275 certificate reconciliation.** The joint solver's
//! `EvidenceJointFitOutcome.fixed_point` is a construction gate, not report
//! telemetry: `false` returns a non-convergence error, so every
//! [`ArrowCofitReport`] necessarily came from an idempotent re-entry.

use ndarray::Array2;

use crate::sparse_dict::BlockChartComposeConfig;

/// Result of the arrow-Schur-routed co-fit linear tier (Stage 1).
#[derive(Clone, Debug)]
pub struct ArrowCofitReport {
    /// Composed reconstruction `N×P` read back from the fitted term.
    pub reconstructed: Array2<f32>,
    /// Explained variance of `reconstructed` against the target (mean-baseline via
    /// the shared `explained_variance_from_reconstruction` helper cofit uses).
    pub explained_variance: f64,
    /// Number of curved (periodic) atoms folded into the joint solve — the count
    /// of blocks whose BIC-gated chart discovery ([`compose_block_coordinate_charts`])
    /// promoted them from a flat linear atom to a curved chart. `0` for the
    /// linear-only path ([`cofit_linear_via_arrow`]). This is the curved-birth
    /// count the migration ledger banks.
    pub n_curved_atoms: usize,
    /// Total BIC complexity charge (`Σ ½·d_eff·ln n_eff`, nats) of the curved
    /// charts folded in — the description-length currency the ledger records as
    /// `dl_bits`. `0.0` for the linear-only path.
    pub curved_charge: f64,
}

/// Tuning for the arrow-routed linear fit. `max_iter` must be generous enough for
/// the evidence policy to settle: a too-small iteration budget is a
/// non-convergence error and can never produce an [`ArrowCofitReport`].
#[derive(Clone, Debug)]
pub struct ArrowCofitConfig {
    pub log_lambda_sparse: f64,
    pub log_lambda_smooth: f64,
    pub max_iter: usize,
    pub step_size: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    /// Number of periodic-harmonic basis columns `M = 2·h + 1` for a folded
    /// curved atom (must be odd, `>= 3`). `3` is one harmonic — an exact circle,
    /// the ring the chart-discovery lane certifies. Used only by
    /// [`cofit_composed_via_arrow`].
    pub curved_num_basis: usize,
    /// Chart-discovery configuration passed to [`compose_block_coordinate_charts`]
    /// to decide WHICH blocks fold in as curved atoms. Its `block_size` /
    /// `block_topk` / `gamma` are overwritten from the passed routing so the tiers
    /// always agree on geometry. Used only by [`cofit_composed_via_arrow`].
    pub chart: BlockChartComposeConfig,
}

impl Default for ArrowCofitConfig {
    fn default() -> Self {
        Self {
            log_lambda_sparse: (1.0e-4f64).ln(),
            log_lambda_smooth: (1.0e-4f64).ln(),
            max_iter: 128,
            step_size: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            curved_num_basis: 3,
            chart: BlockChartComposeConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_dict::{
        BlockSparseConfig, fit_block_sparse_dictionary, reconstruct_block_sparse_rows,
    };
    use ndarray::Array2;

    /// #2023 Inc 5a Stage 1: the arrow-Schur-routed linear tier reproduces the
    /// block-sparse linear reconstruction — its EV must MATCH-OR-BEAT the direct
    /// block reconstruction (same linear model, warm-started from the same routing,
    /// descended by the unified joint solver). Moderate K, dense lane.
    #[test]
    fn arrow_routed_linear_tier_matches_or_beats_block_reconstruction_2023() {
        // Planted linear structure: 3 directions in P=8, K = 3 blocks of b=2.
        let (p, b, n_blocks) = (8usize, 2usize, 3usize);
        let n = 120usize;
        let mut x = Array2::<f32>::zeros((n, p));
        let mut s = 0x2023_5a01u64;
        for i in 0..n {
            for d in 0..n_blocks {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let amp = ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
                x[[i, 2 * d]] += amp;
                x[[i, 2 * d + 1]] += 0.5 * amp;
            }
        }

        let mut config = BlockSparseConfig::new(n_blocks, b);
        config.block_topk = n_blocks;
        config.max_epochs = 60;
        config.aux_k = 2;
        let fit = fit_block_sparse_dictionary(x.view(), &config)
            .expect("block-sparse linear fit must converge on planted structure");

        let block_recon = reconstruct_block_sparse_rows(
            fit.decoder.view(),
            fit.blocks.view(),
            fit.codes.view(),
            b,
        )
        .expect("block reconstruction");
        let block_ev =
            explained_variance_from_reconstruction(x.view(), block_recon.view()).expect("block EV");

        let arrow = cofit_linear_via_arrow(
            x.view(),
            fit.decoder.view(),
            fit.blocks.view(),
            fit.codes.view(),
            fit.gamma,
            &ArrowCofitConfig::default(),
        )
        .expect("arrow-routed linear cofit must run end to end");

        eprintln!(
            "[#2023 5a] block_ev={:.6} arrow_ev={:.6}",
            block_ev, arrow.explained_variance
        );
        assert!(
            arrow.explained_variance.is_finite(),
            "arrow EV must be finite, got {}",
            arrow.explained_variance
        );
        let tol = 1.0e-3 * (1.0 + block_ev.abs());
        assert!(
            arrow.explained_variance >= block_ev - tol,
            "#2023 5a: arrow-routed linear EV {} must match-or-beat block EV {} (tol {})",
            arrow.explained_variance,
            block_ev,
            tol
        );
        assert_eq!(arrow.reconstructed.dim(), (n, p));
    }

    /// #2023 5a idempotence gate — the root cause of the linear/composed cofit
    /// refusals, pinned. A single joint pass from the cold block-decoder seed
    /// descends the penalized objective, so it MOVES state and cannot certify
    /// itself as an idempotent fixed point (the old callers demanded exactly
    /// that ONE cold pass be its own fixed point, and refused). Re-entering the
    /// converged state is a genuine no-op: the solver certifies `fixed_point` and
    /// the fitted reconstruction does not move. This is the term-level evidence
    /// behind [`fit_to_idempotent_reentry_and_read_back`].
    #[test]
    fn arrow_linear_cofit_second_pass_is_a_noop_2023() {
        let (p, b, n_blocks) = (8usize, 2usize, 3usize);
        let n = 120usize;
        let mut x = Array2::<f32>::zeros((n, p));
        let mut s = 0x2023_5a02u64;
        for i in 0..n {
            for d in 0..n_blocks {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                let amp = ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32;
                x[[i, 2 * d]] += amp;
                x[[i, 2 * d + 1]] += 0.5 * amp;
            }
        }
        let mut config = BlockSparseConfig::new(n_blocks, b);
        config.block_topk = n_blocks;
        config.max_epochs = 60;
        config.aux_k = 2;
        let fit = fit_block_sparse_dictionary(x.view(), &config)
            .expect("block-sparse linear fit must converge on planted structure");

        let cofit = ArrowCofitConfig::default();
        let (mut term, mut rho) = build_linear_cofit_term(
            x.view(),
            fit.decoder.view(),
            fit.blocks.view(),
            fit.codes.view(),
            fit.gamma,
            &cofit,
        )
        .expect("build the frozen-support linear cofit term");
        term.set_guards_enabled(false);
        let target = x.mapv(|v| v as f64);

        let joint_pass = |term: &mut SaeManifoldTerm, rho: &mut SaeManifoldRho| {
            term.run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                rho,
                None,
                cofit.max_iter,
                cofit.step_size,
                cofit.ridge_ext_coord,
                cofit.ridge_beta,
            )
            .expect("arrow-Schur joint pass runs")
        };

        // Cold pass: descends from the seed, so it is not its own fixed point.
        let first = joint_pass(&mut term, &mut rho);
        assert!(
            !first.fixed_point,
            "the cold descending pass cannot be its own idempotent fixed point"
        );

        // Re-enter until the converged inner solve certifies (the production loop).
        let mut certified = first.fixed_point;
        for _ in 0..7 {
            if certified {
                break;
            }
            certified = joint_pass(&mut term, &mut rho).fixed_point;
        }
        assert!(
            certified,
            "the linear cofit must reach a certified idempotent fixed point on re-entry"
        );
        let recon_converged = term
            .try_fitted_for_rho(&rho)
            .expect("readback at the fixed point");

        // A further pass over the already-cofit tier is a genuine no-op.
        let extra = joint_pass(&mut term, &mut rho);
        assert!(
            extra.fixed_point,
            "a second cofit pass over the already-cofit linear tier must stay an idempotent no-op"
        );
        let recon_extra = term
            .try_fitted_for_rho(&rho)
            .expect("readback after the no-op re-entry");
        assert_eq!(
            recon_converged, recon_extra,
            "the idempotent re-entry must not move the fitted reconstruction"
        );
    }

    /// Orthonormal-per-block decoder with a planted overlap, mirroring the fixture
    /// `sparse_dict::cofit` is tested on: 3 blocks of size b=2 in P=5, block 1
    /// overlapping block 0 on e1 (so the tied projection double-counts e1), block 2
    /// the plane holding a planted circle.
    fn planted_decoder() -> Array2<f32> {
        let s = 1.0f32 / 2.0f32.sqrt();
        Array2::from_shape_vec(
            (6, 5),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, // e0
                0.0, 1.0, 0.0, 0.0, 0.0, // e1
                0.0, s, s, 0.0, 0.0, // (e1+e2)/√2
                0.0, s, -s, 0.0, 0.0, // (e1−e2)/√2
                0.0, 0.0, 0.0, 1.0, 0.0, // e3
                0.0, 0.0, 0.0, 0.0, 1.0, // e4
            ],
        )
        .unwrap()
    }

    /// Planted linear part in span{e0,e1,e2} plus a unit circle in the {e3,e4}
    /// plane, tiny noise — the trap fixture (a genuine curved chart in block 2).
    fn planted_data(n: usize) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((n, 5));
        for i in 0..n {
            let a = ((i * 7 + 1) % 17) as f32 / 17.0 - 0.5;
            let bb = ((i * 13 + 5) % 19) as f32 / 19.0 - 0.5;
            let cc = ((i * 5 + 3) % 23) as f32 / 23.0 - 0.5;
            let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let noise = 0.002 * (((i * 3) % 11) as f32 / 11.0 - 0.5);
            x[[i, 0]] = a;
            x[[i, 1]] = bb;
            x[[i, 2]] = cc;
            x[[i, 3]] = t.cos() as f32 + noise;
            x[[i, 4]] = t.sin() as f32 - noise;
        }
        x
    }

    /// Tied per-block routing (every row fires all three blocks; γ = 1): exactly
    /// what a converged block-sparse fit stores.
    fn tied_routing(
        x: &Array2<f32>,
        decoder: &Array2<f32>,
        b: usize,
    ) -> (Array2<u32>, Array3<f32>) {
        let n = x.nrows();
        let g = decoder.nrows() / b;
        let mut blocks = Array2::<u32>::zeros((n, g));
        let mut codes = Array3::<f32>::zeros((n, g, b));
        for i in 0..n {
            for gg in 0..g {
                blocks[[i, gg]] = gg as u32;
                for r in 0..b {
                    let atom = decoder.row(gg * b + r);
                    let mut dot = 0.0f32;
                    for c in 0..decoder.ncols() {
                        dot += x[[i, c]] * atom[c];
                    }
                    codes[[i, gg, r]] = dot;
                }
            }
        }
        (blocks, codes)
    }

    fn parity_chart_cfg() -> BlockChartComposeConfig {
        BlockChartComposeConfig {
            block_size: 2,
            block_topk: 3,
            min_firings: 8,
            crossfit_folds: 4,
            pair_screen: false,
            ..BlockChartComposeConfig::default()
        }
    }

    #[test]
    fn insufficient_iterations_return_error_instead_of_open_arrow_cofit_2023() {
        let n = 120usize;
        let b = 2usize;
        let decoder = planted_decoder();
        let x = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, b);
        let config = ArrowCofitConfig {
            max_iter: 0,
            chart: parity_chart_cfg(),
            ..ArrowCofitConfig::default()
        };

        let error = cofit_composed_via_arrow(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &config,
        )
        .expect_err("an open joint-solver re-entry must not mint ArrowCofitReport");
        assert!(
            error.contains("not an idempotent fitted fixed point"),
            "unexpected non-convergence error: {error}"
        );
    }

    /// Re-embed the planted fixture in `p_wide` output dimensions so decoder
    /// FRAMES auto-activate (they need `p >= SAE_FRAME_MIN_AUTO_OUTPUT_DIM`,
    /// and a per-atom decoder rank of 2 shrinks a wide border by far more than
    /// the activation margin). The planted five coordinates keep columns 0..5;
    /// the added columns carry a small independent deterministic signal so no
    /// output column is degenerate and the target keeps full-rank variance.
    /// The decoder is zero on the added columns, so `tied_routing` produces the
    /// IDENTICAL routing and codes as the narrow fixture.
    fn widen_planted_fixture(
        x: &Array2<f32>,
        decoder: &Array2<f32>,
        p_wide: usize,
    ) -> (Array2<f32>, Array2<f32>) {
        let n = x.nrows();
        let p0 = x.ncols();
        assert!(p_wide > p0);
        let mut x_wide = Array2::<f32>::zeros((n, p_wide));
        let mut s = 0x2397_0001u64;
        for i in 0..n {
            for c in 0..p0 {
                x_wide[[i, c]] = x[[i, c]];
            }
            for c in p0..p_wide {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                x_wide[[i, c]] = 0.01 * ((s >> 33) as f32 / (1u32 << 31) as f32 - 1.0);
            }
        }
        let mut d_wide = Array2::<f32>::zeros((decoder.nrows(), p_wide));
        for r in 0..decoder.nrows() {
            for c in 0..p0 {
                d_wide[[r, c]] = decoder[[r, c]];
            }
        }
        (x_wide, d_wide)
    }

    /// #2397 — the composed (curved, and at wide `p` FRAMED) cofit tier's
    /// second pass is a genuine bit-exact no-op, so the whole-pass state
    /// certificate is SOUND on it and needs no gauge quotient.
    ///
    /// #2397 argued the opposite: that `decoder_frame` re-polarization and the
    /// Circle angle re-seed move the model invisibly at fixed objective, so a
    /// "state unchanged" certificate could never certify a framed/curved tier
    /// even at a genuine fixed point — the state would walk the gauge orbit
    /// while the model stood still. The premise does not survive contact with
    /// the movers. Every re-gauge in the joint fit is a class-(c) OBJECTIVE-
    /// GUARDED TRANSACTION over an ALL-OR-NOTHING slice map:
    /// `canonicalize_atom_unit_speed_chart` commits only when the basis absorbs
    /// the reparameterized image to within `CHART_RECOMPOSITION_REL_TOL`, and
    /// REFUSES (`Ok(false)`, atom untouched) otherwise — it never commits the
    /// ε-reslide that would break byte identity. So at a settled state each
    /// re-gauge is either an exact no-op or a genuine improvement, and the two
    /// are distinguishable by the bytes.
    ///
    /// This pins both halves of the claim on the exact production term:
    /// * NARROW (`p = 5`): curved atom, frames off.
    /// * WIDE (`p = 16`): curved atom AND active decoder frames — the tier the
    ///   issue names — so the polar refresh is live in the guarded triple.
    ///
    /// In both, the certified re-entry must recur the raw model state exactly,
    /// the unit-speed retraction driven standalone there must be a no-op, and a
    /// further pass must stay an idempotent no-op with a bit-identical
    /// reconstruction — the #2023 5a linear-tier contract, unchanged, holding
    /// on the curved and framed tiers.
    #[test]
    fn composed_arrow_second_pass_is_a_noop_on_curved_and_framed_tiers_2397() {
        let n = 240usize;
        let b = 2usize;
        let narrow_decoder = planted_decoder();
        let narrow_x = planted_data(n);
        let (wide_x, wide_decoder) = widen_planted_fixture(&narrow_x, &narrow_decoder, 16);

        for (label, x, decoder, expect_frames) in [
            ("narrow", narrow_x.clone(), narrow_decoder.clone(), false),
            ("wide", wide_x, wide_decoder, true),
        ] {
            let (blocks, codes) = tied_routing(&x, &decoder, b);
            let cfg = ArrowCofitConfig {
                max_iter: 256,
                chart: parity_chart_cfg(),
                ..ArrowCofitConfig::default()
            };
            let ComposedCofitTerm {
                mut term,
                mut rho,
                n_curved_atoms,
                ..
            } = build_composed_cofit_term(
                x.view(),
                decoder.view(),
                blocks.view(),
                codes.view(),
                1.0,
                &cfg,
            )
            .expect("build the composed cofit term");
            assert!(
                n_curved_atoms >= 1,
                "[{label}] the gate needs a curved atom in the fold; got {n_curved_atoms}"
            );
            term.set_guards_enabled(false);
            let target = x.mapv(|v| v as f64);

            // Drive the exact production re-entry budget.
            let mut certified_at: Option<usize> = None;
            let mut raw_recurred_at_certification = false;
            for pass in 0..8usize {
                let entry = term.snapshot_mutable_state();
                let outcome = term
                    .run_joint_fit_arrow_schur_for_quasi_laplace(
                        target.view(),
                        &mut rho,
                        None,
                        cfg.max_iter,
                        cfg.step_size,
                        cfg.ridge_ext_coord,
                        cfg.ridge_beta,
                    )
                    .expect("composed joint pass runs");
                let raw_recurred = term.matches_mutable_state(&entry);
                eprintln!(
                    "[#2397 {label}] pass={pass} fixed_point={} raw_state_recurred={raw_recurred} \
                     frames_active={}",
                    outcome.fixed_point,
                    term.frames_active()
                );
                if outcome.fixed_point {
                    certified_at = Some(pass);
                    raw_recurred_at_certification = raw_recurred;
                    break;
                }
            }
            let certified_at = certified_at.unwrap_or_else(|| {
                panic!("[{label}] the composed cofit must reach a certified idempotent fixed point")
            });
            assert_eq!(
                term.frames_active(),
                expect_frames,
                "[{label}] the fixture must exercise the intended frame regime"
            );

            // The certificate's own claim: the pass that certified recurred the
            // RAW model state. This is the assertion #2397 predicted could
            // never hold on a framed/curved tier.
            assert!(
                raw_recurred_at_certification,
                "[{label}] #2397: the certifying re-entry must recur the raw model state \
                 exactly — if this fails the gauge orbit really is being walked and the \
                 certificate needs a quotient (certified at pass {certified_at})"
            );

            // The class-(c) chart re-gauge, driven STANDALONE at the certified
            // state: all-or-nothing, so here it must be nothing.
            let before = term.snapshot_mutable_state();
            let obj_before = term
                .penalized_objective_total(target.view(), &rho, None, 1.0)
                .expect("objective before the standalone retraction");
            let retracted = term
                .retract_unit_speed_charts_in_loop()
                .expect("standalone unit-speed retraction runs");
            let obj_after = term
                .penalized_objective_total(target.view(), &rho, None, 1.0)
                .expect("objective after the standalone retraction");
            assert_eq!(
                retracted, 0,
                "[{label}] #2397: the arc-length slice must be a genuine no-op at the fixed \
                 point, not an ε-reslide that byte identity would resolve as a state move"
            );
            assert!(
                term.matches_mutable_state(&before),
                "[{label}] a zero-atom retraction must leave the state byte-identical"
            );
            assert_eq!(
                obj_before.to_bits(),
                obj_after.to_bits(),
                "[{label}] a no-op re-gauge must not move the penalized objective by one ulp"
            );

            // The #2023 5a second-pass contract, on the curved/framed tier.
            let recon_converged = term
                .try_fitted_for_rho(&rho)
                .expect("readback at the certified fixed point");
            let extra = term
                .run_joint_fit_arrow_schur_for_quasi_laplace(
                    target.view(),
                    &mut rho,
                    None,
                    cfg.max_iter,
                    cfg.step_size,
                    cfg.ridge_ext_coord,
                    cfg.ridge_beta,
                )
                .expect("the extra composed pass runs");
            assert!(
                extra.fixed_point,
                "[{label}] a second pass over the already-cofit composed tier must stay an \
                 idempotent no-op"
            );
            let recon_extra = term
                .try_fitted_for_rho(&rho)
                .expect("readback after the no-op re-entry");
            assert_eq!(
                recon_converged, recon_extra,
                "[{label}] the idempotent re-entry must not move the fitted reconstruction"
            );
        }
    }

    /// #2023 Increment 5 (the composed cutover): the unified arrow-Schur joint solve
    /// over a MIXED linear + curved atom set must MATCH-OR-BEAT the hand-rolled A/B
    /// coordinate-descent co-fit ([`crate::sparse_dict::cofit_block_and_curved`]) in
    /// explained variance on the same planted routing, and it must actually fold in
    /// a curved atom (the discovered circle in block 2). This is the parity evidence
    /// that licenses deleting the alternation and repointing `fit_tiered`.
    #[test]
    fn composed_arrow_matches_or_beats_block_cofit_2023() {
        use crate::sparse_dict::{CofitConfig, cofit_block_and_curved};

        let n = 240usize;
        let b = 2usize;
        let decoder = planted_decoder();
        let x = planted_data(n);
        let (blocks, codes) = tied_routing(&x, &decoder, b);

        // Reference: the hand-rolled A/B co-fit (the deletion target).
        let cofit_cfg = CofitConfig {
            code_ridge: 1.0e-6,
            chart: parity_chart_cfg(),
            ..CofitConfig::default()
        };
        let cofit = cofit_block_and_curved(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &cofit_cfg,
        )
        .expect("block A/B co-fit runs");

        // Candidate: the unified arrow-Schur composed joint solve.
        let arrow_cfg = ArrowCofitConfig {
            max_iter: 256,
            chart: parity_chart_cfg(),
            ..ArrowCofitConfig::default()
        };
        let arrow = cofit_composed_via_arrow(
            x.view(),
            decoder.view(),
            blocks.view(),
            codes.view(),
            1.0,
            &arrow_cfg,
        )
        .expect("composed arrow-Schur co-fit runs end to end");

        eprintln!(
            "[#2023 5] cofit_ev={:.6} arrow_ev={:.6} n_curved={} charge={:.4}",
            cofit.explained_variance,
            arrow.explained_variance,
            arrow.n_curved_atoms,
            arrow.curved_charge
        );

        assert!(
            arrow.explained_variance.is_finite(),
            "composed arrow EV must be finite, got {}",
            arrow.explained_variance
        );
        assert!(
            arrow.n_curved_atoms >= 1,
            "the composed fold must promote at least one curved atom (the circle in \
             block 2); got {}",
            arrow.n_curved_atoms
        );
        let tol = 1.0e-2 * (1.0 + cofit.explained_variance.abs());
        assert!(
            arrow.explained_variance >= cofit.explained_variance - tol,
            "#2023 5: composed arrow EV {} must match-or-beat block A/B co-fit EV {} (tol {})",
            arrow.explained_variance,
            cofit.explained_variance,
            tol
        );
        assert_eq!(arrow.reconstructed.dim(), (n, 5));
    }
}

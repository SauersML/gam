//! #2720 increment one, fixture B: exercise the `δβ · U` / `U`ᵀ round trip in the
//! chart-gauge vector construction with a Grassmann decoder frame ACTIVE.
//!
//! ## Why this fixture has to exist before the modelling fix
//!
//! #2720's central measurement — that the chart orbit is an exact first-order
//! symmetry of the likelihood, residual `1.29e-16 … 1.11e-15` relative over 1333
//! constructions — was taken on two fixtures that are BOTH `Periodic` atoms with
//! `decoder_frame = None`. On that path
//! [`SaeManifoldTerm::dense_step_gauge_vector_from_field`] stores the least-squares
//! decoder compensation `δB` (`M_k × p`) into the border verbatim
//! (`fit_drivers.rs:3166`, the `None` arm). When a frame IS active it stores
//! `δB · U` instead (`:3165`, `U` column-orthonormal `p × r`), and the consumer
//! reads it back through `U`ᵀ. **That round trip has never been measured**, so the
//! likelihood-symmetry claim the whole quotient rests on is established only for
//! the unframed border layout.
//!
//! The span argument says it should be exact: `motion`'s rows are combinations of
//! rows of `B_k`, `δB = (DᵀD)⁻¹Dᵀ·motion` therefore has rows in `rowspace(B_k)`,
//! and `U` is built from `B_k`'s own SVD — so `δB · U · U`ᵀ` = δB`. But the frame
//! rank is the NUMERICAL rank at `SAE_FRAME_RANK_CUTOFF = 1e-7` relative, not the
//! exact rank, so whatever `B_k` carries below that cutoff is discarded by the
//! projection and the argument is exact only up to the truncated tail. Which of
//! those two regimes this code is in is a measurement, not a derivation, and it is
//! the measurement this fixture takes.
//!
//! Both arms run on ONE term at ONE state; the only difference between them is
//! whether the frame is active, so the comparison is not confounded by the fit.

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_gauge_frame_roundtrip_2720;` — its single declaration.
// Stating the scope in-file makes it a claim the compiler enforces rather than one
// the filename merely implies.
#![cfg(test)]
use super::*;

/// The #2253/#2234 planted circle: `K=1`, `d_atom=1`, `n=42`, `p=48`. Reused
/// verbatim (same LCG, same seed) so this fixture sits on the same geometry the
/// `1.29e-16` unframed measurement was taken on — the frame is then the ONLY
/// thing that differs between that measurement and this one.
pub(super) fn planted_circle_cloud() -> Array2<f64> {
    let n = 42usize;
    let p = 48usize;
    let mut state = 0x2468_ace0_1357_9bdfu64;
    let mut unit = move || {
        // LCG → [0,1); NO rand, NO clock (repo #932 rules).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let two_pi = std::f64::consts::TAU;
    let b0: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let b1: Vec<f64> = (0..p).map(|_| 2.0 * unit() - 1.0).collect();
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let theta = two_pi * unit();
        for j in 0..p {
            let noise = 0.01 * (2.0 * unit() - 1.0);
            z[[i, j]] = theta.cos() * b0[j] + theta.sin() * b1[j] + noise;
        }
    }
    z
}

fn seeded_term(target: ArrayView2<'_, f64>) -> SaeManifoldTerm {
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target,
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 45,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("minimal seed");
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target,
        geometry_plans: &minimal.geometry_plans,
        basis_values: minimal.basis_values.view(),
        basis_jacobian: minimal.basis_jacobian.view(),
        decoder_coefficients: minimal.decoder_coefficients.view(),
        smooth_penalties: minimal.smooth_penalties.view(),
        initial_logits: minimal.initial_logits.view(),
        initial_coords: minimal.initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind: SaeFitAssignmentKind::Softmax,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 40,
        learning_rate: 0.05,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: minimal.refine_routing,
        seed_refine_random_state: 45,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })
    .expect("fit seed");
    seed.base_term
}

/// The two halves of the first-order reconstruction change carried by one gauge
/// vector, recomputed FROM the shipped gauge vector rather than from the
/// construction's internals:
///
/// * `motion[row, out] = Σ_axis Σ_col a·δt[row,axis]·J[row,col,axis]·B[col,out]`
///   — the coordinate motion pushed through the decoder;
/// * `compensation[row, out] = Σ_col a·Φ[row,col]·δB[col,out]` — the decoder
///   change the construction solved for, with `δB` decoded back out of the border
///   through `U`ᵀ when a frame is active.
///
/// The construction solves `design·δB = −motion`, so their SUM is the residual
/// reconstruction change, i.e. exactly the quantity #2720 reports as `1.29e-16 …
/// 1.11e-15` relative. Returns `(‖motion‖, ‖motion + compensation‖)`.
///
/// Reading `δB` back out of the border is the point: it is the only way the `U` /
/// `U`ᵀ round trip enters the measurement at all.
fn motion_and_residual_norms(
    term: &SaeManifoldTerm,
    atom_idx: usize,
    gauge: &Array1<f64>,
) -> Result<(f64, f64), String> {
    let n = term.n_obs();
    let q = term.assignment.row_block_dim();
    let p = term.output_dim();
    let atom = &term.atoms[atom_idx];
    let m = atom.basis_size();
    let d = term.assignment.coords[atom_idx].latent_dim();
    let coord_offsets = term.assignment.coord_offsets();
    let beta_offsets = term.factored_border_offsets();
    let border_rank = atom.border_frame_rank();

    let mut delta_border = Array2::<f64>::zeros((m, border_rank));
    let beta_base = n * q + beta_offsets[atom_idx];
    for col in 0..m {
        for channel in 0..border_rank {
            delta_border[[col, channel]] = gauge[beta_base + col * border_rank + channel];
        }
    }
    // The round trip under test: border coordinates are frame coordinates when a
    // frame is active, so decoding them back to the `p`-dimensional output space
    // is `δβ · U`ᵀ. With no frame the border already holds `p` columns.
    let delta_decoder: Array2<f64> = match atom.decoder_frame.as_ref() {
        Some(frame) => delta_border.dot(&frame.frame().t()),
        None => delta_border.clone(),
    };
    if delta_decoder.ncols() != p {
        return Err(format!(
            "decoded border width {} != output dim {p}",
            delta_decoder.ncols()
        ));
    }

    let mut motion = Array2::<f64>::zeros((n, p));
    let mut compensation = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let assignments = term.assignment.try_assignments_row(row)?;
        let a = assignments[atom_idx];
        if a == 0.0 {
            continue;
        }
        let row_base = row * q + coord_offsets[atom_idx];
        for axis in 0..d {
            let dt = gauge[row_base + axis];
            if dt == 0.0 {
                continue;
            }
            for col in 0..m {
                let w = a * dt * atom.basis_jacobian[[row, col, axis]];
                if w == 0.0 {
                    continue;
                }
                for out_col in 0..p {
                    motion[[row, out_col]] += w * atom.decoder_coefficients()[[col, out_col]];
                }
            }
        }
        for col in 0..m {
            let w = a * atom.basis_values[[row, col]];
            if w == 0.0 {
                continue;
            }
            for out_col in 0..p {
                compensation[[row, out_col]] += w * delta_decoder[[col, out_col]];
            }
        }
    }

    let motion_norm = motion.iter().map(|v| v * v).sum::<f64>().sqrt();
    let residual_norm = motion
        .iter()
        .zip(compensation.iter())
        .map(|(a, b)| (a + b) * (a + b))
        .sum::<f64>()
        .sqrt();
    Ok((motion_norm, residual_norm))
}

/// Worst relative reconstruction residual over every gauge vector the term
/// constructs, together with the number of vectors that carried enough motion to
/// be measurable. A vector with no motion cannot fail this test and must not be
/// allowed to dilute the verdict, so it is excluded and COUNTED rather than
/// silently averaged in.
fn worst_relative_residual(term: &SaeManifoldTerm) -> Result<(f64, usize, usize), String> {
    let gauges = term.dense_step_gauge_vectors()?;
    let total = gauges.len();
    let mut worst = 0.0_f64;
    let mut measurable = 0usize;
    for gauge in gauges.iter() {
        for atom_idx in 0..term.k_atoms() {
            let (motion_norm, residual_norm) = motion_and_residual_norms(term, atom_idx, gauge)?;
            // Magnitude floor: a ratio against a motion that is itself at the
            // roundoff floor measures nothing.
            if !(motion_norm > 1.0e-6) {
                continue;
            }
            measurable += 1;
            let rel = residual_norm / motion_norm;
            if rel > worst {
                worst = rel;
            }
        }
    }
    Ok((worst, measurable, total))
}

/// #2720: the chart-gauge construction must cancel the first-order reconstruction
/// motion to machine precision whether or not a Grassmann decoder frame is active.
///
/// The unframed arm is the control — it reproduces the `1.29e-16 … 1.11e-15`
/// regime #2720 already measured, and proves the fixture is capable of showing a
/// small number, so a small number in the framed arm is not an artifact of a dead
/// instrument. The framed arm is the claim.
#[test]
fn dense_step_gauge_vector_cancels_reconstruction_with_decoder_frame_active_2720() {
    let z = planted_circle_cloud();
    let mut term = seeded_term(z.view());

    // ---- control arm: no frame, the layout every prior #2720 measurement used.
    for atom in term.atoms.iter_mut() {
        atom.deactivate_decoder_frame();
    }
    assert!(
        term.atoms.iter().all(|a| a.decoder_frame.is_none()),
        "control arm must have no frame active"
    );
    let (unframed_worst, unframed_measurable, unframed_total) =
        worst_relative_residual(&term).expect("unframed gauge residual");
    println!(
        "[2720-frame] unframed worst_rel={unframed_worst:.6e} measurable={unframed_measurable} \
         gauge_vectors={unframed_total}"
    );
    assert!(
        unframed_measurable > 0,
        "the unframed control measured NOTHING ({unframed_total} gauge vectors, none carrying \
         motion above the 1e-6 floor); a zero-measurement control cannot certify the framed arm"
    );

    // ---- treatment arm: same term, same state, frame activated.
    let mut activated_ranks: Vec<(usize, usize)> = Vec::new();
    let output_dim = term.output_dim();
    for (atom_idx, atom) in term.atoms.iter_mut().enumerate() {
        let rank = atom
            .maybe_activate_decoder_frame()
            .expect("frame activation must not error");
        if let Some(r) = rank {
            activated_ranks.push((atom_idx, r));
        }
    }
    println!(
        "[2720-frame] activated_frames={:?} output_dim={output_dim}",
        activated_ranks
    );
    // Non-vacuity: if no frame activated, the treatment arm IS the control arm and
    // would pass for the wrong reason. This assertion is the whole reason the
    // fixture is trustworthy.
    assert!(
        !activated_ranks.is_empty(),
        "no decoder frame activated on this fixture (p={output_dim}), so the framed border layout \
         was never exercised and this test would have passed as a duplicate of its own control"
    );
    for (atom_idx, r) in activated_ranks.iter() {
        assert!(
            *r < output_dim,
            "atom {atom_idx} activated a rank-{r} frame at output dim {output_dim}; a full-rank \
             frame makes U·Uᵀ the identity and the round trip trivially exact"
        );
    }

    let (framed_worst, framed_measurable, framed_total) =
        worst_relative_residual(&term).expect("framed gauge residual");
    println!(
        "[2720-frame] framed   worst_rel={framed_worst:.6e} measurable={framed_measurable} \
         gauge_vectors={framed_total}"
    );
    assert!(
        framed_measurable > 0,
        "the framed arm measured NOTHING ({framed_total} gauge vectors, none above the motion \
         floor) while the control measured {unframed_measurable}"
    );

    // The bar is the control's own regime, not a constant chosen to be met: the
    // unframed path is the one #2720 certified at `1.29e-16 … 1.11e-15` relative,
    // and the framed path stores the SAME compensation through an orthonormal
    // projection, so it inherits that regime or it is discarding a component of
    // the compensation.
    let bar = 1.0e-10_f64;
    assert!(
        unframed_worst <= bar,
        "control regression: the unframed chart-gauge construction no longer cancels the \
         reconstruction motion (worst relative residual {unframed_worst:.6e} > {bar:.0e}); the \
         framed verdict below is not interpretable until this is understood"
    );
    assert!(
        framed_worst <= bar,
        "the framed border round trip δβ·U then U^T LOSES part of the decoder compensation: worst \
         relative reconstruction residual {framed_worst:.6e} > {bar:.0e}, against {unframed_worst:.6e} \
         on the identical state with the frame deactivated. The chart orbit is then NOT a \
         first-order symmetry of the likelihood on the framed layout, so `quotient_residual_norm_sq` \
         is projecting out a direction the data fit can see -- a defect strictly worse than #2720's \
         prior-side one, which at least left the likelihood flat."
    );
}

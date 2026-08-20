//! #2720 geometry extension: is the chart-gauge orbit-derivative violation
//! Periodic-specific or geometry-general — and does a decoder frame change it?
//!
//! ## Background
//!
//! The issue's "Not established" section records that every #2720 derivative
//! measurement — the original two fixtures, the #2770 re-measurement, and the
//! `b8f0c4d97` Fixture-B round trip — ran on `Periodic` atoms only, and (except
//! for Fixture B's reconstruction-side check) with `decoder_frame = None`.
//! Whether the orbit-tolerance violation survives on flat (`Duchon`) patches,
//! hyperbolic (`Poincare`) patches, or framed atoms is unmeasured.
//!
//! This diagnostic extends the #2770 measurement to those configurations so the
//! modelling fix is designed against the right generality: if the violation is
//! Periodic-specific the fix can live in the periodic gauge construction; if it
//! is geometry-general it must live at the gauge/penalty interface.
//!
//! ## Measured (post-fix, at merged main `5603dec` + this branch, ARD-saddle
//! rho, 40-outer-iteration budget; pre-fix numbers at `9b9973f` in parentheses)
//!
//! | cell | max \|gᵀv\|/tol | worst-dir near-null share (e⁻²⁰) | verdict |
//!|---|---|---|---|
//!| `periodic`/unframed | 0.19× (was 2.33×) | 3.4e-9 | **below — the fix met the criterion** |
//!| `periodic`/framed | 0.19× (was 2.33×) | 3.4e-9 | below (verdict unchanged by frame) |
//!| `duchon`/unframed | 0.01× (was 3.63×) | 5.0e-7 | **below — the fix met the criterion** |
//!| `duchon`/framed | 0.01× (was 3.63×) | 5.0e-7 | below (verdict unchanged by frame) |
//!| `poincare`/unframed | 0.00× (was REFUSED 40 & 400) | 9.6e-4 | **below — first-ever measurement** |
//!| `poincare`/framed | 0.00× (was REFUSED 40) | 9.6e-4 | below (first-ever framed measurement) |
//!| `linear`/unframed | 4.21× (was 0.13×) | 3.4e-9 | **above — the exemption INVERTED** |
//!| `linear`/framed | 4.21× (was 0.13×) | 3.4e-9 | above (verdict unchanged by frame) |
//!
//! What changed between the two tables is the landed #2720/#2762 resolution:
//! `descend_gauge_orbit` (062277d/69ae5a1, 2026-08-15) gives the inner solve a
//! block-coordinate mover for the chart orbit, and ec18eac/61d5f00/44cf16a
//! (2026-08-17) split the single quotient span into
//! `posterior_null_quotient_basis` (decoder nulls only) and `likelihood_flat_block_basis`
//! (what the mover descends). The poincaré REFUSAL — 94.6% of the KKT gradient
//! in-orbit, robust to a 10× budget at `9b9973f` — is GONE: the mover unblocked
//! it (verified by bisection: the refusal dies at `69ae5a1`, before the
//! two-span split, with a bit-identical criterion 8.623218e1).
//!
//! Conclusions the modelling thread can rely on (scoped to this instrument):
//!
//! * **#2720's acceptance criterion is now MET at the exit state for every
//!   curved kind on this fixture** — periodic 0.19×, duchon 0.01×, poincaré
//!   0.00×, all below the solver's own tolerance. The fix lane's fixed-point
//!   test (`at_an_inner_fixed_point_...`, periodic-only, all-zeros rho) is
//!   corroborated at the ARD-saddle penalty state and across geometries.
//! * **The linear exemption INVERTED**: the one kind that was below tolerance
//!   pre-fix (0.13×) is now the only one above it (4.21×). Pre-fix, linear's
//!   exit state genuinely carried little orbit slope; post-fix, the changed
//!   trajectory (orbit-mover consults, once-per-plateau arming) exits at a
//!   state whose dilation direction still carries live slope — recall the
//!   2f766a7 probe: on linear the DILATION field moves the smoothness term by
//!   −7.82 at the seed, the largest orbit slope of any kind. Whether this is
//!   the block descent's once-per-plateau budget running out on the kind with
//!   the steepest orbit, or an exit class that claims a fixed point it does
//!   not have, is the open question this pin now guards: the reading is
//!   "linear exits above orbit tolerance at ARD-saddle rho", not "linear is
//!   broken".
//! * The near-null-penalty control still shows every violation is carried by
//!   the penalty block, not the data term (worst near-null projection
//!   3.4e-5× tol across all cells — asserted), and a decoder frame still
//!   changes no verdict (framed == unframed max ratios, all eight cells).
//!
//! ## Scope and caveats
//!
//! * All cells share one target (the planted circle cloud), which is native
//!   geometry for the periodic atom only: the duchon/linear fits sit at
//!   `data_fit ≈ 130` against periodic's `0.056`. The claim supported is
//!   "a non-periodic kind can violate on this fixture family", not a
//!   magnitude ranking across geometries — a target native to each geometry
//!   would be needed for that.
//! * `poincare`'s pre-fix refusal is GONE post-fix, killed by #2762's
//!   `descend_gauge_orbit` (bisection: the refusal dies at `69ae5a1`, before
//!   the two-span split; the post-fix criterion is bit-identical 8.623218e1
//!   across `69ae5a1` and `5603dec`). The poincaré cells now carry the
//!   first-ever orbit-derivative measurement for the hyperbolic chart. The
//!   escalated-budget retry arm remains as a canary: if the refusal ever
//!   returns at either budget, the pin above fails with this history.
//! * The measurement is a derivative **conditional on fixed ρ** (the
//!   ARD-saddle penalty state), not a statement about the joint posterior
//!   over smoothing parameters: ρ is overridden before the solve, and both
//!   the criterion and the gradient are assembled at that same ρ.
//! * "Below tolerance" for `linear` is a statement about this tolerance
//!   (`SAE_MANIFOLD_INNER_GRAD_REL_TOL · iterate_scale`), not a symmetry
//!   proof.
//!
//! ## Refusal policy
//!
//! Unlike the #2770 baseline (a single known-solvable cell, which `.expect`s
//! its solve), this sweep crosses geometries whose solves may legitimately
//! refuse. Convergence refusal is therefore a REPORTED outcome
//! ([`CellOutcome::Refused`]), while harness failures (seed construction,
//! assembly, frame activation, non-finite values) panic with cell context —
//! a refusal is a measurement about the geometry, a broken instrument is a
//! bug in this file.
//!
//! Enforced (deterministic fixtures): the post-fix findings are asserted —
//! every curved kind (periodic, duchon, poincare) now sits BELOW tolerance,
//! and linear — the pre-fix exempt kind — now sits ABOVE it in its own band.
//! The `#[ignore]`d manual-diagnostic convention is banned by the repo's
//! build-time scanner; this test now runs in the normal suite.
//!
//! ## Method
//!
//! For each atom kind in {`periodic` (control), `duchon`, `poincare`, `linear`}
//! and each frame state in {inactive, active}:
//!
//! 1. Build a seeded term via the minimal-seed path (the same constructor
//!    Fixture B uses), with the atom basis replaced by the kind under test.
//! 2. Override ρ to the ARD-saddle settings BEFORE the solve (the same
//!    settings as the #2770 baseline), so priors are demonstrably active and
//!    the criterion and gradient see one penalty state.
//! 3. Run `penalized_quasi_laplace_criterion_with_cache` to the inner solve
//!    state (40 outer iterations; `poincare` additionally re-attempts at 400).
//! 4. Assemble the Arrow-Schur system, extract the joint KKT gradient.
//! 5. Project onto every chart-gauge direction; report `|gᵀv|/tol` where
//!    `tol = SAE_MANIFOLD_INNER_GRAD_REL_TOL · iterate_scale` — the same gate
//!    as the baseline and the issue's acceptance criteria. Gauge vectors are
//!    unit-normalized by the construction; this test asserts it.
//! 6. Null arm: re-assemble at the same state with penalties collapsed to
//!    `exp(-20)` and project the same directions, so the likelihood-only
//!    share of each violation is visible.

#![cfg(test)]
use super::*;
use crate::manifold::tests_gauge_frame_roundtrip_2720::planted_circle_cloud;

/// One measured cell: the full-penalty sweep and the near-null-penalty
/// controls at the same state.
#[derive(Debug, Clone, Copy)]
struct CellMeasurement {
    max_ratio: f64,
    directions: usize,
    tolerance: f64,
    /// For the direction with the worst total ratio: `|g_nullᵀv| / |gᵀv|` at
    /// penalty floor `exp(-20)` — the near-null share of THAT direction.
    null_share_worst_dir: f64,
    /// Same, at floor `exp(-26)`. If the near-null projection is residual
    /// penalty floor (not likelihood leakage) it shrinks by ≈ e⁻⁶ here.
    null_share_worst_dir_deep: f64,
    /// Worst `|g_nullᵀv| / tol` over ALL directions at floor `exp(-20)` —
    /// asserted small per direction, so contamination cannot hide in a weak
    /// direction.
    max_null_over_tol: f64,
}

/// The outcome of one cell: either a measurement, or a convergence refusal
/// (reported, never interpreted as a symmetry statement). Harness failures
/// panic; they are not outcomes.
#[derive(Debug, Clone)]
enum CellOutcome {
    Measured(CellMeasurement),
    Refused { budget: usize, note: String },
}

/// Build a seeded one-atom term of the requested basis kind on the planted
/// circle cloud — the identical seed path Fixture B (`b8f0c4d97`) uses, with
/// only the `atom_basis` token changed. Returns the term and its seed rho.
fn seeded_term_of_kind(
    kind: &str,
    target: ArrayView2<'_, f64>,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target,
        atom_basis: vec![kind.to_string()],
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
    .unwrap_or_else(|e| panic!("[2720-geom] minimal seed failed for {kind}: {e}"));
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
    .unwrap_or_else(|e| panic!("[2720-geom] fit seed failed for {kind}: {e}"));
    (seed.base_term, seed.initial_rho)
}

/// Apply the #2770 baseline's ard-saddle penalty settings to the seed rho.
/// Applied BEFORE the solve; the criterion and every gradient in this file
/// are assembled at this same rho (fixed-ρ derivative — see module docs).
fn ard_saddle_rho(mut rho: SaeManifoldRho) -> SaeManifoldRho {
    rho.log_lambda_sparse = -0.5;
    for value in rho.log_lambda_smooth.iter_mut() {
        *value = -1.0;
    }
    for axis in rho.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = -0.5;
        }
    }
    rho
}

/// Collapse every penalty channel to `exp(log_floor)` (≈ 2e-9 at −20) while
/// preserving the rho LAYOUT (per-atom smooth vector, ARD axis shapes,
/// block/kappa entries), so the near-null arm assembles a system of identical
/// shape at the same state. Two floors (`−20`, `−26`) distinguish residual
/// penalty floor from likelihood leakage: a residual-floor projection scales
/// with `exp(floor)`, leakage does not.
fn near_null_penalty_rho(rho: &SaeManifoldRho, log_floor: f64) -> SaeManifoldRho {
    let mut null = rho.clone();
    null.log_lambda_sparse = log_floor;
    for value in null.log_lambda_smooth.iter_mut() {
        *value = log_floor;
    }
    for axis in null.log_ard.iter_mut() {
        for value in axis.iter_mut() {
            *value = log_floor;
        }
    }
    for value in null.log_lambda_block.iter_mut() {
        *value = log_floor;
    }
    null
}

/// Joint KKT gradient of the Arrow-Schur system at the given penalty state,
/// with the row-offset layout the gauge basis expects.
fn joint_gradient(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    label: &str,
) -> (Array1<f64>, usize, usize, Vec<usize>) {
    let sys = term
        .assemble_arrow_schur(target, rho, None)
        .unwrap_or_else(|e| panic!("[2720-geom] {label}: assemble_arrow_schur failed: {e}"));
    let n_rows = sys.rows.len();
    let border_dim = sys.gb.len();
    let full_len: usize = sys.rows.iter().map(|r| r.gt.len()).sum::<usize>() + border_dim;
    let mut grad = Array1::<f64>::zeros(full_len);
    let mut row_offsets = Vec::with_capacity(n_rows + 1);
    row_offsets.push(0usize);
    let mut offset = 0usize;
    for row in &sys.rows {
        for (i, &v) in row.gt.iter().enumerate() {
            grad[offset + i] = v;
        }
        offset += row.gt.len();
        row_offsets.push(offset);
    }
    for (i, &v) in sys.gb.iter().enumerate() {
        grad[offset + i] = v;
    }
    assert!(
        grad.iter().all(|v| v.is_finite()),
        "[2720-geom] {label}: KKT gradient contains non-finite entries"
    );
    (grad, n_rows, border_dim, row_offsets)
}

/// The #2770 measurement core with the null-penalty control: inner solve →
/// Arrow-Schur → joint gauge basis → `|gᵀv|/tol` per direction, plus the
/// likelihood-only share via a penalties-collapsed re-assembly at the same
/// state. Returns [`CellOutcome::Refused`] only for convergence refusal.
fn measure_orbit_projection_2720(
    term: &mut SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    budget: usize,
    label: &str,
) -> CellOutcome {
    let result = term.penalized_quasi_laplace_criterion_with_cache(
        target, rho, None, budget, 0.4, 1.0e-6, 1.0e-6,
    );
    let (criterion_value, loss, _cache) = match result {
        Ok(ok) => ok,
        Err(e) => {
            // Only the RECOGNIZED convergence-refusal is a Refused outcome;
            // anything else (NaN, malformed state, gate bugs) is an
            // instrument failure and must fail loudly with context.
            let note = e.to_string();
            assert!(
                note.contains("inner solve did not converge"),
                "[2720-geom] {label}: inner solve failed for an unrecognized reason \
                 (not the known convergence refusal): {note}"
            );
            // Validate the refusal telemetry rather than trusting it: a
            // null_share outside [0,1] or non-finite refusal norms mean the
            // refusal message itself is corrupt. The field was renamed
            // `gauge_share` -> `null_share` when the chart orbit left the
            // convergence quotient (a7aad87); parse either so a refusal on
            // either side of that rename still gets validated.
            let share_token = ["null_share=", "gauge_share="].iter().find_map(|key| {
                note.find(key).map(|idx| {
                    note[idx + key.len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e')
                        .collect::<String>()
                })
            });
            if let Some(token) = share_token {
                if let Ok(share) = token.parse::<f64>() {
                    assert!(
                        share.is_finite() && (0.0..=1.0).contains(&share),
                        "[2720-geom] {label}: refusal telemetry corrupt — \
                         gauge_share={share} outside [0,1]"
                    );
                    eprintln!(
                        "[2720-geom] {label}: refusal telemetry: {note} \
                         (from a NONSTATIONARY iterate; not an orbit measurement)"
                    );
                }
            }
            return CellOutcome::Refused { budget, note };
        }
    };
    eprintln!(
        "[2720-geom] {label}: criterion={criterion_value:.6e} \
         (data_fit={:.3e}, sparsity={:.3e}, smoothness={:.3e}, ard={:.3e})",
        loss.data_fit, loss.assignment_sparsity, loss.smoothness, loss.ard
    );

    let (grad, n_rows, border_dim, row_offsets) = joint_gradient(term, target, rho, label);
    let grad_norm = grad.dot(&grad).sqrt();
    eprintln!("[2720-geom] {label}: n_rows={n_rows} border_dim={border_dim} ‖g‖={grad_norm:.6e}");

    let gauge_basis = term
        .joint_chart_gauge_basis_for_arrow_layout(
            &row_offsets,
            border_dim,
            &format!("2720-geom {label}"),
        )
        .unwrap_or_else(|e| panic!("[2720-geom] {label}: joint_chart_gauge_basis failed: {e}"));
    if gauge_basis.is_empty() {
        panic!(
            "[2720-geom] {label}: NO gauge directions at this state — the measurement \
             instrument is inapplicable here, which for these fixtures is a harness bug \
             (the baseline measured ≥1 direction on the same fixture family)"
        );
    }

    // The construction unit-normalizes (fit_drivers MGS); assert what the
    // ratio below relies on rather than trusting it silently.
    for (i, v) in gauge_basis.iter().enumerate() {
        let norm = v.dot(v).sqrt();
        assert!(
            (norm - 1.0).abs() < 1.0e-12,
            "[2720-geom] {label}: gauge vector v_{i} has norm {norm:.6e}, expected 1"
        );
    }

    let iterate_scale = term.inner_iterate_scale();
    let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * iterate_scale;
    assert!(
        tolerance.is_finite() && tolerance > 0.0,
        "[2720-geom] {label}: tolerance is non-finite or non-positive"
    );
    // NOTE on the gate's norm: the solver's own stationarity gate is L2
    // (‖g‖ ≤ tol, construction_quasi_laplace.rs:931) with tol defined exactly
    // as above. With unit v, |gᵀv| ≤ ‖g‖₂, so any direction whose projection
    // exceeds tolerance provably violates the solver's L2 gate — no
    // componentwise ‖v‖₁ factor applies. Reported ratios are therefore
    // conservative: the full gradient norm is at least as large.

    // Per-direction bookkeeping: total ratio, and near-null projections at
    // two penalty floors so contamination cannot hide in a weak direction and
    // residual floor can be distinguished from likelihood leakage.
    let null_grad = {
        let null_rho = near_null_penalty_rho(rho, -20.0);
        let (g, _, _, _) = joint_gradient(term, target, &null_rho, label);
        g
    };
    let deep_grad = {
        let deep_rho = near_null_penalty_rho(rho, -26.0);
        let (g, _, _, _) = joint_gradient(term, target, &deep_rho, label);
        g
    };

    let mut max_ratio = 0.0f64;
    let mut worst_abs = 0.0f64;
    let mut worst_dir = 0usize;
    let mut max_null_over_tol = 0.0f64;
    for (i, v) in gauge_basis.iter().enumerate() {
        let proj = grad.dot(v);
        assert!(
            proj.is_finite(),
            "[2720-geom] {label}: v_{i} projection is non-finite"
        );
        let ratio = proj.abs() / tolerance;
        let null_proj = null_grad.dot(v).abs();
        let null_over_tol = null_proj / tolerance;
        max_null_over_tol = max_null_over_tol.max(null_over_tol);
        if proj.abs() > worst_abs {
            worst_abs = proj.abs();
            worst_dir = i;
        }
        max_ratio = max_ratio.max(ratio);
        eprintln!(
            "[2720-geom] {label}: v_{i}: |gᵀv| = {:.6e}  ({:.2}× tolerance)  \
             near-null e⁻²⁰: {:.3e}  ({:.2e}× tol)",
            proj.abs(),
            ratio,
            null_proj,
            null_over_tol
        );
    }

    // Per-direction near-null gate: EVERY direction's near-null projection
    // must sit far below tolerance, else the likelihood term itself carries
    // gauge content on this fixture (which would falsify #2720's premise).
    assert!(
        max_null_over_tol < 1.0e-3,
        "[2720-geom] {label}: near-null projection hit {:.3e}× tol on some direction — \
         the data term carries gauge content, so the violation is NOT purely prior-side",
        max_null_over_tol
    );

    let null_share_worst_dir = null_grad.dot(&gauge_basis[worst_dir]).abs() / worst_abs;
    let null_share_worst_dir_deep = deep_grad.dot(&gauge_basis[worst_dir]).abs() / worst_abs;
    eprintln!(
        "[2720-geom] {label}: max |gᵀv|/tol = {max_ratio:.2}×  over {} directions  \
         (tol={tolerance:.3e}; worst-dir near-null share: {:.2e} at e⁻²⁰, {:.2e} at e⁻²⁶)",
        gauge_basis.len(),
        null_share_worst_dir,
        null_share_worst_dir_deep
    );

    CellOutcome::Measured(CellMeasurement {
        max_ratio,
        directions: gauge_basis.len(),
        tolerance,
        null_share_worst_dir,
        null_share_worst_dir_deep,
        max_null_over_tol,
    })
}

/// One (kind, frame) cell of the geometry sweep.
fn run_cell_2720(kind: &str, framed: bool) -> CellOutcome {
    let label = format!("{kind}/{}", if framed { "framed" } else { "unframed" });
    let z = planted_circle_cloud();
    let (mut term, rho) = seeded_term_of_kind(kind, z.view());
    let rho = ard_saddle_rho(rho);

    if framed {
        let output_dim = term.output_dim();
        let mut activated = Vec::new();
        for (atom_idx, atom) in term.atoms.iter_mut().enumerate() {
            match atom.maybe_activate_decoder_frame() {
                Ok(Some(rank)) => activated.push((atom_idx, rank)),
                Ok(None) => {}
                Err(e) => panic!("[2720-geom] {label}: frame activation errored: {e}"),
            }
        }
        eprintln!("[2720-geom] {label}: activated_frames={activated:?} output_dim={output_dim}");
        assert!(
            !activated.is_empty(),
            "[2720-geom] {label}: no decoder frame activated, so the framed cell would \
             silently duplicate the unframed cell"
        );
        for (_atom_idx, rank) in activated {
            assert!(
                rank < term.output_dim(),
                "[2720-geom] {label}: rank-{rank} frame at output_dim {} is full rank; \
                 the round trip is trivially exact and measures nothing",
                term.output_dim()
            );
        }
    } else {
        for atom in term.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
    }

    measure_orbit_projection_2720(&mut term, z.view(), &rho, 40, &label)
}

/// #2720 geometry sweep: the orbit-derivative measurement across atom kinds
/// and frame states. See module docs. The periodic/unframed cell is the
/// instrument control.
#[test]
fn chart_gauge_orbit_violation_across_geometries_2720() {
    let kinds = ["periodic", "duchon", "poincare", "linear"];

    eprintln!("[2720-geom] ════════ geometry × frame sweep ════════");
    let mut summary: Vec<(String, CellOutcome)> = Vec::new();
    for kind in kinds {
        for framed in [false, true] {
            let label = format!("{kind}/{}", if framed { "framed" } else { "unframed" });
            let cell = run_cell_2720(kind, framed);
            summary.push((label, cell));
        }
    }

    eprintln!("[2720-geom] ════════ summary ════════");
    eprintln!("[2720-geom] (tol gate = SAE_MANIFOLD_INNER_GRAD_REL_TOL · iterate_scale)");
    for (label, cell) in &summary {
        match cell {
            CellOutcome::Measured(m) => {
                let verdict = if m.max_ratio <= 1.0 {
                    "AT/BELOW tol"
                } else {
                    "ABOVE tol"
                };
                eprintln!(
                    "[2720-geom] {label:<20} max|gᵀv|/tol = {:6.2}×  ({} dirs, tol={:.3e}, \
                     near-null: {:.1e}/e⁻²⁰ {:.1e}/e⁻²⁶ worst-dir, {:.1e}× tol max)  {verdict}",
                    m.max_ratio,
                    m.directions,
                    m.tolerance,
                    m.null_share_worst_dir,
                    m.null_share_worst_dir_deep,
                    m.max_null_over_tol
                );
            }
            CellOutcome::Refused { budget, note } => eprintln!(
                "[2720-geom] {label:<20} REFUSED under budget={budget}: {}",
                note.chars().take(160).collect::<String>()
            ),
        }
    }

    // Instrument health: every cell must have been attempted, and every
    // non-poincare kind must have produced a MEASUREMENT in both frames — a
    // refusal there means the instrument broke, not the geometry refused.
    assert_eq!(
        summary.len(),
        8,
        "the sweep must attempt all 8 cells (4 kinds × 2 frame states)"
    );
    for (label, cell) in &summary {
        if !label.starts_with("poincare") {
            assert!(
                matches!(cell, CellOutcome::Measured(_)),
                "[2720-geom] {label}: expected a measurement — a refusal outside poincare \
                 is an instrument failure, not a geometry result"
            );
        }
    }
    let control = &summary[0];
    assert!(
        matches!(control.1, CellOutcome::Measured(_)),
        "[2720-geom] the periodic/unframed control produced no measurement; every \
         other cell is uninterpretable until the instrument control works"
    );

    // Poincare budget escalation: if the 40-budget arms refused, re-attempt at
    // 400 so the refusal is established as robust (or overturned).
    let poincare_refused = summary
        .iter()
        .filter(|(l, c)| l.starts_with("poincare") && matches!(c, CellOutcome::Refused { .. }))
        .count();
    if poincare_refused > 0 {
        eprintln!("[2720-geom] ════════ poincare escalated-budget retry (400) ════════");
        let z = planted_circle_cloud();
        let (mut term, rho) = seeded_term_of_kind("poincare", z.view());
        let rho = ard_saddle_rho(rho);
        for atom in term.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        match measure_orbit_projection_2720(&mut term, z.view(), &rho, 400, "poincare/retry400") {
            CellOutcome::Measured(m) => eprintln!(
                "[2720-geom] poincare/retry400      max|gᵀv|/tol = {:6.2}× over {} dirs — \
                 the 40-budget refusal was budget-conditioned, measurement obtained",
                m.max_ratio, m.directions
            ),
            CellOutcome::Refused { budget, note } => eprintln!(
                "[2720-geom] poincare/retry400      STILL REFUSED under budget={budget}: {}",
                note.chars().take(160).collect::<String>()
            ),
        }
    }

    // ENFORCED findings (deterministic fixtures; re-measured post-fix at
    // merged main `5603dec`; pre-fix values at `9b9973f` in the module docs):
    //   1. the fix met the criterion for every curved kind — periodic (was
    //      2.33×) and duchon (was 3.63×) must now sit BELOW tolerance;
    //   2. poincare no longer refuses — the #2762 orbit mover unblocked the
    //      stall (bisection: refusal dies at `69ae5a1`), so the cell must now
    //      MEASURE, and its measurement must sit below tolerance;
    //   3. linear is now the ONE kind above tolerance (was 0.13× below): the
    //      exemption inverted. Pinned in its own band so the finding cannot
    //      silently drift; the reading is "linear exits above orbit tolerance
    //      at ARD-saddle rho", not "linear is broken";
    //   4. framing changes no verdict (framed == unframed max-ratio bands).
    let measured = |label: &str| {
        summary
            .iter()
            .find(|(l, _)| l == label)
            .unwrap_or_else(|| panic!("[2720-geom] cell {label} missing from summary"))
    };
    for kind in ["periodic", "duchon", "poincare"] {
        for frame in ["unframed", "framed"] {
            let (label, cell) = measured(&format!("{kind}/{frame}"));
            let m = match cell {
                CellOutcome::Measured(m) => *m,
                CellOutcome::Refused { .. } => panic!(
                    "[2720-geom] {label} refused — the three curved kinds measured at both \
                     frames post-fix (the pre-fix poincare refusal was killed by #2762's \
                     descend_gauge_orbit); a refusal now is an instrument or solver change"
                ),
            };
            assert!(
                m.max_ratio < 1.0,
                "[2720-geom] {label}: max|gᵀv|/tol = {:.2}× is above tolerance — the \
                 post-fix finding is that the landed #2720/#2762 resolution meets the \
                 issue's acceptance criterion on every curved kind (measured 0.19× / \
                 0.01× / 0.00×); if this rose, the orbit mover lost the curved kinds",
                m.max_ratio
            );
        }
    }
    for frame in ["unframed", "framed"] {
        let (label, cell) = measured(&format!("linear/{frame}"));
        let m = match cell {
            CellOutcome::Measured(m) => *m,
            CellOutcome::Refused { .. } => panic!(
                "[2720-geom] {label} refused — linear measured at both frames when this \
                 pin was re-taken; a refusal now is an instrument or solver change"
            ),
        };
        assert!(
            m.max_ratio > 1.0 && m.max_ratio < 20.0,
            "[2720-geom] {label}: max|gᵀv|/tol = {:.2}× left the post-fix band (1, 20) \
             — the linear exemption INVERTED in the post-fix world (pre-fix 0.13× below, \
             post-fix 4.21× above); if this band fails the linear exit story changed; \
             re-measure before updating",
            m.max_ratio
        );
    }
}

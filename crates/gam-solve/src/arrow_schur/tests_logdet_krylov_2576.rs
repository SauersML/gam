//! #2576 — what the evidence lane's shifted-CG ladder actually does, measured
//! rather than argued.
//!
//! The issue's headline evidence is that loosening the CG tolerance from `1e-8`
//! to `1e-4` did not shorten the log-determinant solve, read as proof that CG
//! stagnates. That reading is not forced: an identical iteration count at two
//! tolerances four decades apart is equally consistent with
//!
//! * (a) an ill-conditioned operator whose CG crawls — the residual curve is
//!   nearly flat and the Ritz spectrum is wide;
//! * (b) a non-symmetric operator, for which CG is simply the wrong algorithm
//!   and no preconditioner helps;
//! * (c) loss of orthogonality, where the recursive residual and the true
//!   residual part company and the recurrence has to be restarted;
//! * (d) a solve that converges FAST, where the last four decades of tolerance
//!   cost a handful of iterations and the cost is somewhere else entirely.
//!
//! These need opposite repairs, and nothing in the crate could tell them apart,
//! because a solve reported one summed iteration count and nothing else. The
//! instrument here reports the residual curve, the Ritz spectrum read off the CG
//! coefficients at no extra matvec, the symmetry defect, and the per-node
//! breakdown of the shift ladder.

use super::*;
use gam_linalg::utils::splitmix64;

/// A Gauss–Newton arrow system with an overcomplete, heavy-tailed atom
/// dictionary — the shape #2576 is about.
///
/// Every existing SAE fixture in this crate installs a SYNTHETIC shared block
/// (`sae_structured_system`: `hbb[[r, r]] = k + 4`, unrelated to the cross-block
/// it will be compared against), and this issue's own history records that the
/// choice, not the operator, decided the answer twice. This builds the honest
/// object instead: per row a residual Jacobian `J_i = [A_i | B_i]` with
/// `B_i = L̃_i P_i` sparse over the row's active atoms, and
///
/// ```text
/// H_tt^(i) = A_iᵀA_i,   H_tβ^(i) = A_iᵀ L̃_i P_i,   H_ββ = Σ_i P_iᵀ L̃_iᵀ L̃_i P_i.
/// ```
///
/// Two properties follow that no synthetic block can give:
///
/// * `S = H_ββ − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` is the Schur complement of a
///   PSD matrix, hence PSD — SPD once the caller's `ridge_beta` is added — with
///   NO diagonal-dominance fudge holding it up;
/// * the eliminated term is the projection of `L̃_i` onto each row's own latent
///   directions, so it is a LARGE share of the border rather than the ≲2%
///   correction the synthetic fixtures produce. That share is what makes `S`
///   ill-conditioned, and reproducing it is the whole point.
///
/// Atom usage is drawn heavy-tailed (`atom ∝ u^tail_exponent`), which is the
/// firing-count distribution the issue names as the border's conditioning:
/// atoms appearing in a handful of rows next to atoms appearing in thousands.
///
/// Returns the system and the per-row support, deterministic in `seed`.
fn overcomplete_gauss_newton_arrow(
    rows: usize,
    latent: usize,
    channels: usize,
    residual: usize,
    atoms: usize,
    top_k: usize,
    tail_exponent: f64,
    seed: u64,
) -> (ArrowSchurSystem, Vec<Vec<(usize, f64)>>) {
    let k = atoms * channels;
    let mut sys = ArrowSchurSystem::new(rows, latent, k);
    let mut state = seed | 1;
    let unit = |state: &mut u64| -> f64 {
        let bits = splitmix64(state) >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    };
    let normal = |state: &mut u64| -> f64 {
        let u1 = unit(state).max(1e-12);
        let u2 = unit(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut support: Vec<Vec<(usize, f64)>> = Vec::with_capacity(rows);
    let mut local_jac: Vec<Vec<f64>> = Vec::with_capacity(rows);
    let mut hbb = Array2::<f64>::zeros((k, k));

    for row in 0..rows {
        // A_i: residual × latent. B̃_i (`l_tilde`): residual × channels.
        let mut a = Array2::<f64>::zeros((residual, latent));
        for value in a.iter_mut() {
            *value = normal(&mut state);
        }
        let mut l_tilde = Array2::<f64>::zeros((residual, channels));
        for value in l_tilde.iter_mut() {
            *value = normal(&mut state);
        }
        // H_tt = AᵀA (SPD whenever residual ≥ latent and A has full column rank,
        // which a Gaussian draw gives with probability one).
        sys.rows[row].htt = a.t().dot(&a);
        sys.rows[row].gt = Array1::<f64>::zeros(latent);
        // The fixture's `L_i` in the residency's sense is AᵀL̃ (latent × channels,
        // row-major), so `H_tβ^(i) = L_i P_i` exactly as the SAE operator assumes.
        let cross = a.t().dot(&l_tilde);
        let mut jac = vec![0.0_f64; latent * channels];
        for c in 0..latent {
            for j in 0..channels {
                jac[c * channels + j] = cross[[c, j]];
            }
        }
        local_jac.push(jac);

        // Heavy-tailed atom draw; duplicates within a row are merged so the
        // projector coefficient on a column is the SUM of its support entries,
        // matching what `resident_schur_elimination_diagonal` prices.
        let mut active: Vec<(usize, f64)> = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            let u = unit(&mut state);
            let atom = ((atoms as f64) * u.powf(tail_exponent)) as usize;
            let atom = atom.min(atoms - 1);
            let phi = 0.5 + unit(&mut state);
            let base = atom * channels;
            match active.iter_mut().find(|(seen, _)| *seen == base) {
                Some(entry) => entry.1 += phi,
                None => active.push((base, phi)),
            }
        }
        // H_ββ += P_iᵀ L̃_iᵀ L̃_i P_i — the full dense p×p gram on every support
        // PAIR, not just the diagonal. Dropping the cross terms would break the
        // PSD-by-construction property the whole fixture rests on.
        let gram = l_tilde.t().dot(&l_tilde);
        for &(base_a, phi_a) in active.iter() {
            for &(base_b, phi_b) in active.iter() {
                let scale = phi_a * phi_b;
                for r in 0..channels {
                    for c in 0..channels {
                        hbb[[base_a + r, base_b + c]] += scale * gram[[r, c]];
                    }
                }
            }
        }
        support.push(active);
    }
    sys.hbb = hbb;
    sys.gb = Array1::<f64>::zeros(k);

    let forward_support = support.clone();
    let forward_jac = local_jac.clone();
    let transpose_support = support.clone();
    let transpose_jac = local_jac.clone();
    let p = channels;
    sys.set_row_htbeta_operator(
        move |row, x, out| {
            let mut gathered = vec![0.0_f64; p];
            for &(base, phi) in &forward_support[row] {
                for j in 0..p {
                    gathered[j] += phi * x[base + j];
                }
            }
            let jac = &forward_jac[row];
            let latent = jac.len() / p;
            for c in 0..latent {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += jac[c * p + j] * gathered[j];
                }
                out[c] = acc;
            }
        },
        move |row, v, out| {
            let jac = &transpose_jac[row];
            let latent = jac.len() / p;
            let mut scattered = vec![0.0_f64; p];
            for c in 0..latent {
                let vc = v[c];
                for j in 0..p {
                    scattered[j] += jac[c * p + j] * vc;
                }
            }
            for &(base, phi) in &transpose_support[row] {
                for j in 0..p {
                    out[base + j] += phi * scattered[j];
                }
            }
        },
    );
    sys.set_device_sae_pcg_data(DeviceSaePcgData {
        p: channels,
        beta_dim: k,
        a_phi: std::sync::Arc::from(support.clone().into_boxed_slice()),
        local_jac: std::sync::Arc::from(local_jac.into_boxed_slice()),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: None,
    });
    sys.refresh_row_hessian_fingerprint();
    (sys, support)
}

/// Emit the residual curve, Ritz spectrum and per-node ladder profile of one
/// evidence-lane log-determinant evaluation, and pin the two facts that decide
/// which #2576 fault class this is.
///
/// The assertions here are deliberately about the INSTRUMENT's validity, not
/// about the verdict: a fixture that could not express a wide spectrum, or an
/// operator that was not symmetric, would make every number below meaningless.
/// The verdict is read off the printed curve.
#[test]
fn shift_ladder_residual_curve_and_ritz_spectrum_2576() {
    let (rows, latent, channels, residual, atoms, top_k) = (4_000, 3, 8, 5, 96, 4);
    let ridge_beta = 1e-6;
    let (sys, support) = overcomplete_gauss_newton_arrow(
        rows, latent, channels, residual, atoms, top_k, 4.0, 0x2576_A6A6,
    );
    let k = sys.k;

    // Non-vacuity of the fixture: the firing-count distribution must actually be
    // heavy-tailed, or the "wide border diagonal" this issue is about is absent
    // and no conclusion drawn here transfers.
    let mut firing = vec![0usize; atoms];
    for row in support.iter() {
        for &(base, _) in row.iter() {
            firing[base / channels] += 1;
        }
    }
    let hot = firing.iter().copied().max().unwrap_or(0);
    let cold = firing.iter().copied().filter(|c| *c > 0).min().unwrap_or(0);
    let diagonal = sys.shared_block_diagonal();
    let diag_hi = diagonal.iter().copied().fold(0.0_f64, f64::max);
    let diag_lo = diagonal
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "#2576 fixture: k={k} atoms={atoms} rows={rows} | firing counts {cold}..{hot} \
         ({:.1}x) | diag(H_bb) {diag_lo:.4e}..{diag_hi:.4e} ({:.1}x)",
        hot as f64 / cold.max(1) as f64,
        diag_hi / diag_lo,
    );
    assert!(
        hot as f64 / cold.max(1) as f64 >= 10.0,
        "#2576: the fixture must carry a heavy-tailed firing count, else it \
         cannot express the conditioning this issue is about (hot {hot}, cold {cold})"
    );

    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, latent, false)
        .expect("Gauss-Newton per-row blocks are SPD and must factor");
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend);

    let profile = reduced_schur_logdet_shift_ladder_profile(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        resident.as_ref(),
        4,
        0x2576_5CD1,
        1.0e-8,
        60,
        1.0e-8,
        20_000,
    )
    .expect("the profiled evaluation must run on an SPD Gauss-Newton arrow system");

    eprintln!(
        "#2576 ladder: nodes={} total_iters={} log|S|={:+.9e} +- {:.3e} | \
         bracket [{:.4e}, {:.4e}] | symmetry defect {:.3e} | concentration {:.2}x",
        profile.nodes.len(),
        profile.total_iterations,
        profile.log_det,
        profile.std_err,
        profile.bracket.0,
        profile.bracket.1,
        profile.symmetry_defect,
        profile.ladder_concentration(),
    );

    // (b): CG is only a valid algorithm on a symmetric operator. If this were
    // large, every other number here would be describing the wrong question.
    assert!(
        profile.symmetry_defect < 1.0e-10,
        "#2576: the reduced Schur apply must be symmetric for CG to be the right \
         algorithm at all (defect {:.3e})",
        profile.symmetry_defect
    );

    eprintln!("#2576 NODE_TABLE pos shift weight iters max_iters kappa");
    for node in profile.nodes.iter() {
        eprintln!(
            "#2576 NODE {} {:.9e} {:.9e} {} {} {}",
            node.ladder_position,
            node.shift,
            node.weight,
            node.iterations,
            node.max_solve_iterations,
            node.krylov_condition
                .map(|kappa| format!("{kappa:.6e}"))
                .unwrap_or_else(|| "nan".to_string()),
        );
    }

    let hardest = &profile.hardest_solve;
    eprintln!(
        "#2576 HARDEST shift={:.9e} iters={} certified={} kappa={:?}",
        hardest.shift,
        hardest.iterations(),
        hardest.certified,
        hardest.krylov_condition_estimate(),
    );
    for (label, trace) in [
        ("COLD", &profile.cold_seed_solve),
        ("COLDRAW", &profile.cold_seed_solve_undiagonalized),
    ] {
        eprintln!(
            "#2576 {label} shift={:.9e} iters={} certified={} kappa={:?} bound={:?}",
            trace.shift,
            trace.iterations(),
            trace.certified,
            trace.krylov_condition_estimate(),
            trace.conditioning_iteration_bound(),
        );
        for (step, value) in trace.relative_residuals().iter().enumerate() {
            eprintln!("#2576 RESID_{label} {step} {value:.9e}");
        }
        let ritz = trace.ritz_values(trace.iterations().min(64));
        for (index, value) in ritz.iter().enumerate() {
            eprintln!("#2576 RITZ_{label} {index} {value:.9e}");
        }
    }
    eprintln!(
        "#2576 BUDGET one_krylov_space={:?} measured_per_rhs={:.1}",
        profile.one_krylov_space_apply_budget(),
        profile.total_iterations as f64 / 4.0,
    );

    // The instrument must have resolved something: a trace with fewer than two
    // usable Lanczos coefficients reports no spectrum, and every claim about the
    // curve above would be vacuous.
    assert!(
        hardest.iterations() >= 2,
        "#2576: the hardest solve must take at least two steps for the Ritz \
         spectrum to exist (took {})",
        hardest.iterations()
    );
    assert!(
        profile.log_det.is_finite(),
        "#2576: the profiled evaluation must produce the surrogate value"
    );
}

/// #2576 — the quadrature ladder must cost ONE Krylov space per right-hand side,
/// not one per quadrature node.
///
/// # What this pins, and why it is not an iteration budget
///
/// A shift adds a multiple of the identity, so
/// `K_m(S + t_ℓ I, v) = K_m(S, v)` for every node: the ladder's `L` systems are
/// `L` projections onto ONE subspace. Serving them from one Krylov space costs
/// that space plus one certification apply per node, and the space's own size is
/// set by the operator's conditioning through the standard CG bound. So the
/// acceptance is
///
/// ```text
/// applies per right-hand side  ≤  ⌈½·√κ·ln(2/rel_tol)⌉ + node_count
/// ```
///
/// where `κ` is read off the Ritz values of the cold seed solve on THIS operator
/// and `node_count` is the plan's own. Nothing in the bound is chosen; a
/// better-conditioned operator or a coarser quadrature moves it automatically,
/// and it is not satisfiable by capping an iteration count, because an
/// uncertified iterate is refused upstream.
///
/// # The two failure directions
///
/// * A ladder that rebuilds a Krylov space per node blows the budget — that is
///   the pre-#2576 evidence lane, and the control arm below measures it in the
///   same process on the same plan so the comparison is not a memory of another
///   run.
/// * A family evaluator that met the budget by returning less-converged solves
///   would move the value; both arms are therefore checked against the EXACT
///   dense `log|S|`, which this fixture is small enough to form.
#[test]
fn evidence_logdet_ladder_costs_one_krylov_space_per_probe_2576() {
    let (rows, latent, channels, residual, atoms, top_k) = (4_000, 3, 8, 5, 96, 4);
    let ridge_beta = 1e-6;
    let probes = 8usize;
    let seed = 0x2576_5CD1_u64;
    let (rel_tol, power_iters, cg_rel_tol, cg_max_iters) = (1.0e-8, 60usize, 1.0e-8, 20_000usize);
    let (sys, _support) = overcomplete_gauss_newton_arrow(
        rows, latent, channels, residual, atoms, top_k, 4.0, 0x2576_A6A6,
    );
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, latent, false)
        .expect("Gauss-Newton per-row blocks are SPD and must factor");
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend);

    // The oracle: the dense reduced Schur this whole module exists to avoid
    // forming, formed here once because the fixture is small enough that it can
    // be, so "made it fast" is checked against truth rather than against the
    // other arm's agreement.
    let dense = build_dense_schur_direct(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
    .expect("dense reduced Schur must build at this fixture size");
    let cholesky = cholesky_lower(&dense).expect("the Gauss-Newton reduced Schur is SPD");
    let exact_log_det: f64 = (0..sys.k).map(|i| 2.0 * cholesky[[i, i]].ln()).sum();

    // Control arm: the per-shift ladder, measured on this operator in this
    // process, with the per-node breakdown and the cold seed's conditioning.
    let profile = reduced_schur_logdet_shift_ladder_profile(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        resident.as_ref(),
        probes,
        seed,
        rel_tol,
        power_iters,
        cg_rel_tol,
        cg_max_iters,
    )
    .expect("the per-shift control ladder must run");
    let budget = profile
        .one_krylov_space_apply_budget()
        .expect("the cold seed must resolve a Ritz spectrum to derive a bound from");
    let control_per_rhs = profile.total_iterations as f64 / probes as f64;

    // Production arm: the same plan, the same operator, the same tolerances,
    // through the entry point the evidence lane actually calls.
    let (plan, eval) = rational_reduced_schur_log_det(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        resident.as_ref(),
        None,
        probes,
        seed,
        rel_tol,
        power_iters,
        cg_rel_tol,
        cg_max_iters,
    )
    .expect("the production rational log-determinant must evaluate");
    let family_per_rhs = eval.cg_iterations as f64 / probes as f64;

    eprintln!(
        "#2576 ACCEPT nodes={} kappa_seed_shared={:?} kappa_seed_raw={:?} \
         budget/rhs={budget:.1} | per-shift control {control_per_rhs:.1} applies/rhs | \
         family {family_per_rhs:.1} applies/rhs | exact log|S| {exact_log_det:+.9e} | \
         control {:+.9e} +- {:.3e} | family {:+.9e} +- {:.3e}",
        plan.nodes.len(),
        profile.cold_seed_solve.krylov_condition_estimate(),
        profile
            .cold_seed_solve_undiagonalized
            .krylov_condition_estimate(),
        profile.log_det,
        profile.std_err,
        eval.estimate,
        eval.std_err,
    );

    // Non-vacuity: the bound must be one the per-shift ladder actually misses,
    // or passing it says nothing. This is the measurement that makes the
    // acceptance a gate rather than a formality.
    assert!(
        control_per_rhs > budget,
        "#2576: the per-shift control ladder must exceed the one-Krylov-space \
         budget, else this fixture cannot distinguish the two ladders \
         (control {control_per_rhs:.1}, budget {budget:.1})"
    );

    assert!(
        family_per_rhs <= budget,
        "#2576: the evidence ladder must serve all {} quadrature nodes from ONE \
         Krylov space per right-hand side. Its own conditioning allows \
         {budget:.1} applies per right-hand side (CG bound at the cold seed's \
         measured kappa, plus one certification apply per node); it spent \
         {family_per_rhs:.1}.",
        plan.nodes.len()
    );

    // ... and it is the SAME number. Both arms carry their own Hutchinson bar;
    // a 5-sigma band on the smaller of the two plus the quadrature's own
    // `rel_tol` budget is the honest tolerance for an estimator being compared
    // with the truth it estimates.
    let bar = profile.std_err.min(eval.std_err);
    let allowance = 5.0 * bar + rel_tol * exact_log_det.abs().max(1.0);
    for (label, value) in [("per-shift", profile.log_det), ("family", eval.estimate)] {
        let gap = (value - exact_log_det).abs();
        assert!(
            gap <= allowance,
            "#2576: the {label} ladder's log|S| {value:+.12e} is {gap:.3e} from the \
             exact {exact_log_det:+.12e}, outside the 5-sigma + quadrature \
             allowance {allowance:.3e} — this is a fast answer, not a wrong one"
        );
    }
    let arm_gap = (profile.log_det - eval.estimate).abs();
    assert!(
        arm_gap <= allowance,
        "#2576: the two ladders evaluate ONE functional and must agree far inside \
         its error bar (gap {arm_gap:.3e}, allowance {allowance:.3e})"
    );
}

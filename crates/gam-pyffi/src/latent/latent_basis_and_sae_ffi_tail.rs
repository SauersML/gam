/// Build [`SaeAtomBuildPlan`]s from `(z, atom_basis, atom_dim)` + per-atom
/// PCA seed. Periodic atoms get `n_harmonics = max(1, d_atom)`; Duchon atoms
/// get deterministic center indices from the PCA seed.
fn sae_build_atom_plans(
    z: ArrayView2<'_, f64>,
    atom_basis: &[String],
    atom_dim: &[usize],
    seed_coords: ArrayView3<'_, f64>,
    random_state: u64,
) -> Result<Vec<SaeAtomBuildPlan>, String> {
    let k_atoms = atom_basis.len();
    let n_obs = z.nrows();
    let seed_shape = seed_coords.shape();
    if atom_dim.len() != k_atoms {
        return Err(format!(
            "sae_build_atom_plans: atom_dim length {} must equal atom_basis length {k_atoms}",
            atom_dim.len()
        ));
    }
    if seed_shape[0] != k_atoms || seed_shape[1] != n_obs {
        return Err(format!(
            "sae_build_atom_plans: seed_coords must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            seed_shape
        ));
    }
    let mut plans: Vec<SaeAtomBuildPlan> = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let kind = sae_atom_basis_kind_from_str(&atom_basis[atom_idx]);
        let d = atom_dim[atom_idx];
        if d == 0 {
            return Err(format!(
                "sae_build_atom_plans: atom_dim[{atom_idx}] must be positive"
            ));
        }
        if d > seed_shape[2] {
            return Err(format!(
                "sae_build_atom_plans: atom_dim[{atom_idx}]={d} exceeds seed_coords D_max={}",
                seed_shape[2]
            ));
        }
        match &kind {
            SaeAtomBasisKind::Periodic => {
                // A periodic atom parameterises a circle, which is intrinsically
                // 1-dimensional: the latent coordinate is a single phase angle
                // `t ∈ [0, 1)`. The user-facing `atom_dim` (a.k.a. `d_atom`) for
                // a periodic basis selects the *number of harmonics* in the
                // truncated Fourier expansion (basis size `2·n_harmonics + 1`),
                // not a latent-space dimensionality. Setting
                // `latent_dim = atom_dim` would make
                // `build_sae_basis_evaluators` reject the atom (the analytic
                // `PeriodicHarmonicEvaluator` requires `latent_dim == 1`),
                // since there is no longer a frozen-snapshot fallback. Bind the
                // optimizer-visible latent dimension to 1 and route the user's
                // `d_atom` into the harmonic count.
                let n_harmonics = d.max(1);
                let basis_size = sae_periodic_basis_size(n_harmonics)?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Periodic,
                    latent_dim: 1,
                    n_harmonics,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Sphere => {
                // The (lat, lon) chart fixes latent_dim = 2 and basis_size = 7
                // regardless of the user-supplied `atom_dim` — the chart
                // already captures the embedded S² geometry. Reject any
                // d_atom other than 2 to keep the contract honest.
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'sphere' requires atom_dim == 2, got {d}"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Sphere,
                    latent_dim: 2,
                    n_harmonics: 0,
                    duchon_centers: None,
                    basis_size: SAE_SPHERE_BASIS_SIZE,
                });
            }
            SaeAtomBasisKind::Torus => {
                // Torus of dim `d` uses a tensor-product periodic harmonic
                // basis of size `(2H+1)^d`. The user's `atom_dim` selects
                // the latent dimension; `n_harmonics` defaults to
                // `SAE_DEFAULT_TORUS_HARMONICS`. The design grows
                // exponentially in `d`, so reject runaway combinations.
                let h = SAE_DEFAULT_TORUS_HARMONICS;
                let evaluator = TorusHarmonicEvaluator::new(d, h)?;
                let basis_size = evaluator.basis_size();
                if basis_size > SAE_MAX_PERIODIC_HARMONICS * 4 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} torus basis size {basis_size} = (2*{h}+1)^{d} exceeds the dense limit; reduce atom_dim or harmonics"
                    ));
                }
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Torus,
                    latent_dim: d,
                    n_harmonics: h,
                    duchon_centers: None,
                    basis_size,
                });
            }
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::Duchon
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                // A Duchon atom's curvature penalty degrades (and ultimately
                // fails its D2 collocation) when the center count does not
                // exceed the polynomial nullspace dimension of its resolved
                // order. Pick enough centers to clear that dimension with a
                // margin (so a positive-rank kernel block survives), bounded
                // above by `n_obs` and the dense cap. The Euclidean patch
                // ignores centers, so this lower bound is harmless there.
                let duchon_m = sae_duchon_atom_m(d);
                let poly_nullspace_dim = duchon_nullspace_dimension(d, duchon_m.saturating_sub(1));
                let center_floor = (poly_nullspace_dim + d + 1).max(8);
                let center_ceiling = center_floor.max(32);
                let lo = center_floor.min(n_obs);
                let hi = center_ceiling.min(n_obs);
                let n_centers = n_obs.min(hi).max(lo);
                let idx = sae_pick_duchon_center_indices(
                    n_obs,
                    n_centers,
                    random_state.wrapping_add(atom_idx as u64),
                );
                let mut centers = Array2::<f64>::zeros((idx.len(), d));
                for (out_row, src_row) in idx.iter().copied().enumerate() {
                    for col in 0..d {
                        centers[[out_row, col]] = seed_coords[[atom_idx, src_row, col]];
                    }
                }
                // Probe one build to learn the final basis size. The linear atom
                // builds the degree-1 monomial patch `{1, t}` (width `d + 1`); the
                // euclidean (quadratic) patch builds the degree-2 monomial patch
                // (#1221); everything else (Duchon) uses the thin-plate kernel.
                let probe_pts = Array2::<f64>::zeros((1, d));
                let (phi, _jet, _penalty) = match kind {
                    SaeAtomBasisKind::Linear => {
                        sae_build_euclidean_atom_with_degree(probe_pts.view(), centers.view(), 1)?
                    }
                    SaeAtomBasisKind::EuclideanPatch | SaeAtomBasisKind::Poincare => {
                        sae_build_euclidean_atom(probe_pts.view(), centers.view())?
                    }
                    _ => sae_build_duchon_atom(probe_pts.view(), centers.view())?,
                };
                let basis_size = phi.ncols();
                plans.push(SaeAtomBuildPlan {
                    kind,
                    latent_dim: d,
                    n_harmonics: 0,
                    duchon_centers: Some(centers),
                    basis_size,
                });
            }
            SaeAtomBasisKind::Mobius => {
                // Möbius band (#2240) is a first-class SEEDABLE kind: the
                // deck-invariant double-cover layout is fixed by the production
                // convention, so the plan needs no centers — just the width.
                if d != 2 {
                    return Err(format!(
                        "sae_build_atom_plans: atom {atom_idx} basis 'mobius' requires atom_dim == 2, got {d}"
                    ));
                }
                let evaluator = MobiusHarmonicEvaluator::new(
                    SAE_MOBIUS_CIRCLE_HARMONICS,
                    SAE_MOBIUS_WIDTH_DEGREE,
                )?;
                plans.push(SaeAtomBuildPlan {
                    kind: SaeAtomBasisKind::Mobius,
                    latent_dim: 2,
                    n_harmonics: SAE_MOBIUS_CIRCLE_HARMONICS,
                    duchon_centers: None,
                    basis_size: evaluator.basis_size(),
                });
            }
            SaeAtomBasisKind::Cylinder => {
                // A cylinder atom is not SEEDED through `sae_manifold_fit_minimal`:
                // it arises only by EVIDENCE, when the #977 birth topology race
                // selects `S¹ × ℝ` for a residual factor (the born atom's evaluator
                // is built directly by `race_birth_topology`, and OOS refresh reads
                // it back through `build_sae_basis_evaluators`). There is no
                // user-facing cylinder seed geometry to derive a plan from here, so
                // a cylinder in the seed dictionary is a caller error, surfaced
                // loudly rather than mis-built as a torus / patch.
                return Err(
                    "sae_build_atom_plans: 'cylinder' is a birth-discovered topology, not a seed \
                     dictionary kind; seed with periodic, duchon, sphere, torus, or \
                     euclidean_patch and let the structure search grow a cylinder by evidence"
                        .to_string(),
                );
            }
            SaeAtomBasisKind::FiniteSet => {
                // The finite-set (discrete-anchor) candidate is inert scaffolding
                // not enrolled in the topology race by default
                // (`structure_harvest::finite_set_race_enrolled` is false), and its
                // latent is CATEGORICAL rather than a continuous seed coordinate, so
                // there is no user-facing finite-set seed geometry to derive a plan
                // from here. First-class integration into the continuous-latent
                // optimizer is the documented follow-up; until then a finite_set in
                // the seed dictionary is a caller error, surfaced loudly rather than
                // mis-built as a patch.
                return Err(
                    "sae_build_atom_plans: 'finite_set' is a discrete-anchor (categorical) \
                     candidate that is not enrolled in the topology race and cannot be seeded \
                     through sae_manifold_fit_minimal; seed with periodic, duchon, sphere, \
                     torus, or euclidean_patch"
                        .to_string(),
                );
            }
            SaeAtomBasisKind::Precomputed(name) => {
                return Err(format!(
                    "sae_build_atom_plans: unsupported atom_basis {:?}; sae_manifold_fit_minimal can build only periodic, duchon, sphere, torus, or euclidean_patch atoms",
                    name
                ));
            }
        }
    }
    Ok(plans)
}

/// One-shot SAE-manifold fit driver: takes only `(z, atom_basis, atom_dim,
/// ...scalar hyperparams)` and assembles the full basis + jacobian + penalty
/// stack + PCA seed coords + zero-init decoder + zero-init logits internally
/// before delegating to the same end-to-end Rust Newton loop as
/// [`sae_manifold_fit`]. Returns the same payload dict with one extra key,
/// `"atom_plans"`, holding the per-atom basis spec so OOS prediction can
/// rebuild the design without going through Python.
///
/// Warm starts (issue #357): `initial_logits` (N, K) and `initial_coords`
/// (K, N, D_max) are optional caller-supplied seeds for the assignment logits
/// and the per-atom on-manifold coordinates. When supplied they replace the
/// internal PCA seed coords / zero-jitter logit init, so an amortized encoder
/// can predict `(a_init, t_init)` and have the joint solver refine them for a
/// bounded `max_iter` steps. The basis *design* (Duchon centers, harmonic
/// counts) is still derived from the PCA seed so the warm coordinates are
/// evaluated against the same atom geometry the unconstrained fit would build.
#[pyfunction(signature = (
    z,
    atom_basis,
    atom_dim,
    alpha,
    tau,
    learnable_alpha,
    assignment_kind,
    sparsity_strength = 1.0,
    smoothness = 1.0,
    max_iter = 50,
    learning_rate = 0.05,
    ridge_ext_coord = 1.0e-6,
    ridge_beta = 1.0e-6,
    gumbel_schedule = None,
    analytic_penalties = None,
    random_state = 0,
    top_k = None,
    initial_logits = None,
    initial_coords = None,
    jumprelu_threshold = 0.0,
    native_ard_enabled = true,
    fisher_factors = None,
    fisher_mass_residual = None,
    fisher_provenance = None,
    row_loss_weights = None,
    separation_barrier_strength_override = None,
    ibp_alpha_override = None,
    structured_residual_passes = 2,
    // #2239 magic-by-default: evidence-certified residual structure is promoted
    // to the primary tier by default (the certificate gates the birth; the
    // alternation self-extends its pass budget only while lineages are live).
    promote_from_residual = true,
    run_structure_search = true,
    run_outer_rho_search = true,
    // #2228/#1095/#2132 — the SCALE-gauge is DEFAULT-ON to cure the decoder-penalty
    // ↔ gate co-collapse (mirrors the `sae_manifold_fit_inner` default; the d06c1255f
    // flip missed THIS entry, which is the one the Python `sae_manifold_fit` facade
    // routes through). Callers can pass `quotient_scale=False` for a historical A/B.
    quotient_scale = true,
    data_row_reseed = false,
    // #1893: default Python auto fits to the realised-rank REML/Laplace
    // complexity ledger; callers can set false for historical A/B.
    rank_charge_evidence = true,
))]
fn sae_manifold_fit_minimal<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    alpha: f64,
    tau: f64,
    learnable_alpha: bool,
    assignment_kind: String,
    sparsity_strength: f64,
    smoothness: f64,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    gumbel_schedule: Option<&Bound<'py, PyDict>>,
    analytic_penalties: Option<String>,
    random_state: u64,
    top_k: Option<usize>,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    jumprelu_threshold: f64,
    native_ard_enabled: bool,
    // WP-D output-Fisher shard (#980). `(n, p, r)` f64 factors; presence activates
    // `RowMetric::OutputFisher`. This is the entry point the high-level Python
    // `sae_manifold_fit` facade routes through, so it carries the shard the same
    // magic-by-default way as the precomputed-basis `sae_manifold_fit`.
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_mass_residual: Option<PyReadonlyArray1<'py, f64>>,
    // Harvest provenance tag (#980): same-position `"output_fisher"` (default) or
    // forward-looking `"output_fisher_downstream"`. Routed to the matching
    // `RowMetric` constructor; gauge/lens/dose consume either unchanged.
    fisher_provenance: Option<String>,
    // Per-row design-honesty reconstruction weights (#977); `(n,)` √w. Absent ⇒
    // unweighted path. Installed on the term before the joint fit / ρ selection.
    row_loss_weights: Option<PyReadonlyArray1<'py, f64>>,
    // Per-fit config (separation-barrier strength / IBP-α). `Some` pins this
    // term's value; `None` selects the canonical data-derived or mode default.
    separation_barrier_strength_override: Option<f64>,
    ibp_alpha_override: Option<f64>,
    // #2021 — count of extra whitened-residual structured-alternation passes.
    // Default-ON at `2` ("magic by default"); pass `0` for the historical
    // iid-only path, bit-for-bit.
    structured_residual_passes: usize,
    promote_from_residual: bool,
    run_structure_search: bool,
    run_outer_rho_search: bool,
    quotient_scale: bool,
    data_row_reseed: bool,
    rank_charge_evidence: bool,
) -> PyResult<Py<PyDict>> {
    // #1777 — accept both "threshold_gate" (primary) and legacy "jumprelu".
    let assignment_kind = canonicalize_assignment_kind(&assignment_kind).map_err(py_value_error)?;
    let z_view = z.as_array();
    let (n_obs, _p_out) = z_view.dim();
    let k_atoms = atom_basis.len();
    if n_obs == 0 || z_view.ncols() == 0 {
        return Err(py_value_error(format!(
            "sae_manifold_fit_minimal: z must be non-empty; got shape ({}, {})",
            n_obs,
            z_view.ncols()
        )));
    }
    if k_atoms == 0 {
        return Err(py_value_error(
            "sae_manifold_fit_minimal: atom_basis must be non-empty".into(),
        ));
    }
    // Front-door enforcement through the single shared seam (#985 / E1): the dense
    // manifold engine is the small-K CERTIFICATION lane for penalty-gated modes,
    // whose N×K logits are live Newton state. The hard TOP-K SUPPORT mode carries
    // no gate coordinates, so its admission is the CONCRETE in-core memory budget
    // (`admit_topk_manifold`): within budget the TRUE manifold engine runs at any
    // overcompleteness K > P; over budget it refuses with a typed error — a topk
    // manifold request is never silently substituted with the linear lane.
    if assignment_kind == "topk" {
        let support = top_k.ok_or_else(|| {
            py_value_error(
                "sae_manifold_fit_minimal: assignment_kind 'topk' requires the top_k \
                 argument (the fixed per-row support size)"
                    .to_string(),
            )
        })?;
        let d_max = atom_dim.iter().copied().max().unwrap_or(1);
        gam::terms::sae::front_door::admit_topk_manifold(
            n_obs,
            z_view.ncols(),
            k_atoms,
            d_max,
            support,
        )
        .map_err(py_value_error)?;
    } else {
        gam::terms::sae::front_door::admit_dense_certification(n_obs, z_view.ncols(), k_atoms)
            .map_err(py_value_error)?;
    }
    if !z_view.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_fit_minimal: z contains non-finite values".into(),
        ));
    }
    if atom_dim.len() != k_atoms {
        return Err(py_value_error(format!(
            "sae_manifold_fit_minimal: atom_dim length {} must equal atom_basis length {k_atoms}",
            atom_dim.len()
        )));
    }
    // #2238/#2239 — per-atom topology discovery for the PRIMARY dictionary.
    // Atoms seeded "auto" (the magic default when the caller names no
    // topology) are rewritten to their evidence-race winners BEFORE any
    // seeding or plan building, so every downstream consumer (PCA seeds,
    // atom plans, OOS metadata) sees only concrete kinds. The policy lives
    // in gam-sae (`resolve_auto_primary_atoms`); this layer only plumbs.
    let mut atom_basis = atom_basis;
    let mut atom_dim = atom_dim;
    if atom_basis.iter().any(|basis| basis == "auto") {
        let labels = sae_output_energy_cluster_labels(z_view, k_atoms);
        gam::terms::sae::structure_harvest::resolve_auto_primary_atoms(
            z_view,
            &labels,
            &mut atom_basis,
            &mut atom_dim,
        )?;
    }
    let basis_kinds: Vec<SaeAtomBasisKind> = atom_basis
        .iter()
        .map(|kind| sae_atom_basis_kind_from_str(kind))
        .collect();
    let seed_coords =
        gam::terms::sae::manifold::sae_pca_seed_initial_coords(z_view, &basis_kinds, &atom_dim)
            .map_err(py_value_error)?;
    let plans = sae_build_atom_plans(
        z_view,
        &atom_basis,
        &atom_dim,
        seed_coords.view(),
        random_state,
    )
    .map_err(py_value_error)?;
    // The optimizer's per-atom latent dimension is `plan.latent_dim`, not the
    // user-supplied `atom_dim` (periodic atoms carry a harmonic count there).
    let plan_latent_dim: Vec<usize> = plans.iter().map(|plan| plan.latent_dim).collect();
    // Warm-start coordinates (issue #357). When the caller supplies
    // `initial_coords` (an amortized encoder's predicted on-manifold `t`), use
    // them as the Newton start and as the point at which the basis stacks are
    // first evaluated; otherwise fall back to the PCA seed. The basis *design*
    // (`plans`) is always built from the PCA seed so the atom geometry matches
    // the cold-start fit. Shape must be `(K, N, D_max)` with `D_max` covering
    // every atom's `plan.latent_dim`.
    let mut start_coords: Array3<f64> = match &initial_coords {
        Some(arr) => {
            let view = arr.as_array();
            let shape = view.shape();
            if shape.len() != 3 {
                return Err(py_value_error(
                    "sae_manifold_fit_minimal: initial_coords must be a rank-3 (K, N, D_max) array"
                        .to_string(),
                ));
            }
            if shape[0] != k_atoms || shape[1] != n_obs {
                return Err(py_value_error(format!(
                    "sae_manifold_fit_minimal: initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {shape:?}"
                )));
            }
            let max_dim = *shape.get(2).unwrap_or(&0);
            for (atom_idx, &d) in plan_latent_dim.iter().enumerate() {
                if d > max_dim {
                    return Err(py_value_error(format!(
                        "sae_manifold_fit_minimal: initial_coords D_max={max_dim} is too small for atom {atom_idx} latent_dim={d}"
                    )));
                }
            }
            if !view.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_manifold_fit_minimal: initial_coords contains non-finite values".into(),
                ));
            }
            view.to_owned()
        }
        None => seed_coords.clone(),
    };
    if initial_coords.is_none()
        && k_atoms > 1
        && matches!(assignment_kind.as_str(), "softmax" | "ibp_map")
    {
        let labels = sae_output_energy_cluster_labels(z_view, k_atoms);
        let plan_kinds: Vec<SaeAtomBasisKind> =
            plans.iter().map(|plan| plan.kind.clone()).collect();
        sae_refine_periodic_seed_coords_by_cluster(z_view, &plan_kinds, &labels, &mut start_coords)
            .map_err(py_value_error)?;
    }
    let (basis_values, basis_jacobian, smooth_penalties, basis_sizes, _coord_blocks) =
        sae_build_padded_basis_stacks(&plans, start_coords.view(), n_obs)
            .map_err(py_value_error)?;
    // The JumpReLU gate activates only when a logit clears
    // `jumprelu_threshold` (caller-configurable, default 0.0). Seeding the
    // logits at or below the threshold would leave every gate closed at step
    // 0, making the data-fit Jacobian, the sparsity prior gradient, and the
    // assignment-weighted decoder gradient all zero simultaneously — the fit
    // cannot escape that fixed point. Seed JumpReLU runs a fixed margin
    // ABOVE the configured threshold so every atom starts active relative to
    // its cut and the fit can learn which atoms to prune. Softmax
    // (translation-invariant) remains neutral at zero, while IBP-MAP uses the
    // zero seed except for its degenerate K=1 gate handled below.
    // Warm-start logits (issue #357): a caller-supplied `(N, K)` assignment
    // logit seed (from an amortized encoder) replaces the cold-start init.
    // When absent we fall back to the documented zero / JumpReLU-positive init
    // plus the seed-keyed jitter below.
    let warm_logits: Option<Array2<f64>> = match &initial_logits {
        Some(arr) => {
            let view = arr.as_array();
            if view.dim() != (n_obs, k_atoms) {
                return Err(py_value_error(format!(
                    "sae_manifold_fit_minimal: initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
                    view.dim()
                )));
            }
            if !view.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_manifold_fit_minimal: initial_logits contains non-finite values".into(),
                ));
            }
            Some(view.to_owned())
        }
        None => None,
    };
    let logits_are_cold = warm_logits.is_none();
    let mut initial_logits = match warm_logits {
        Some(logits) => logits,
        None if assignment_kind == "threshold_gate" => {
            // Start every atom one full margin above its activation threshold.
            const SAE_JUMPRELU_SEED_MARGIN: f64 = 1.0;
            Array2::<f64>::from_elem(
                (n_obs, k_atoms),
                jumprelu_threshold + SAE_JUMPRELU_SEED_MARGIN,
            )
        }
        None if k_atoms == 1 && assignment_kind == "ibp_map" => {
            // At K=1 the IBP stick-breaking prior is degenerate (pi_0 == 1), so the
            // gate zeta = sigma(logit/tau) is a free multiplicative scalar on the
            // reconstruction with no competing atom and no sparsity pressure. A zero
            // seed starts it at sigma(0)=0.5 -- a 50% radial seed contraction the
            // joint fit must climb back from against a vanishing sigmoid gradient,
            // landing the ring inside the data (#1023). Seed the single atom
            // "present" so zeta starts ~1; the gate stays free to fall if the atom is
            // genuinely vacuous (the post-fit EV collapse guard, not zeta->0, flags
            // that, so part-3 collapse detection is unaffected). Temperature-robust:
            // seed logit = c*tau so zeta = sigma(c) is independent of tau.
            const SAE_IBP_K1_PRESENT_GATE_LOGIT: f64 = 6.0;
            Array2::<f64>::from_elem((n_obs, k_atoms), SAE_IBP_K1_PRESENT_GATE_LOGIT * tau)
        }
        None => Array2::<f64>::zeros((n_obs, k_atoms)),
    };
    // Data-driven asymmetric cold-start seed (issue #629). A uniform logit
    // init is an exact symmetric saddle for K>=2 exchangeable atoms under
    // softmax / IBP-MAP, so the fit never routes and the decoder overfits
    // through the frozen uniform mixture. Replace the saddle with one EM-style
    // M-then-E step on the seed geometry: prefer the atom that best
    // reconstructs each row. JumpReLU keeps its margin-above-threshold seed
    // (its degeneracy is a closed gate, not a routing tie), and warm-started
    // logits are left exactly as supplied.
    if logits_are_cold && k_atoms > 1 && matches!(assignment_kind.as_str(), "softmax" | "ibp_map") {
        const SAE_RESIDUAL_SEED_GAIN: f64 = 4.0;
        let residual_logits = sae_residual_seed_logits(
            basis_values.view(),
            &basis_sizes,
            z_view,
            SAE_RESIDUAL_SEED_GAIN,
        )
        .map_err(py_value_error)?;
        initial_logits = residual_logits;
    }
    // Wire `random_state` into the optimizer init: jitter the initial
    // assignment logits with a tiny, seed-keyed deterministic perturbation
    // so different seeds explore different Newton trajectories (issue #178).
    // The jitter is uniform in `±SAE_RANDOM_STATE_LOGIT_JITTER` and uses the
    // same Lehmer LCG pattern as the Duchon-center picker, keyed by
    // `random_state`. Fixed seeds still produce bit-identical fits. A
    // warm-started logit seed is left untouched: the encoder's prediction is
    // the requested starting point and the bounded refinement must begin
    // exactly there.
    if n_obs > 0 && k_atoms > 0 && logits_are_cold {
        const SAE_RANDOM_STATE_LOGIT_JITTER: f64 = 1.0e-3;
        let mut state = random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for row in 0..n_obs {
            for atom_idx in 0..k_atoms {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map top 53 bits to a double in [0, 1), then to [-1, 1).
                let u = ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
                let signed = 2.0 * u - 1.0;
                initial_logits[[row, atom_idx]] += SAE_RANDOM_STATE_LOGIT_JITTER * signed;
            }
        }
    }
    // Seed each atom's decoder block by least-squares projection of Z onto
    // the joint atom design weighted by the iter-0 soft assignments. Without
    // this, multi-atom fits collapse to A=0 because the assignment prior
    // dominates a zero-decoder fit — see [`sae_decoder_lsq_init`] for the
    // full diagnosis (issue #174).
    let decoder_coefficients = sae_decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        z_view,
        initial_logits.view(),
        assignment_kind.as_str(),
        ibp_alpha_override.unwrap_or(alpha),
        tau,
        jumprelu_threshold,
        top_k,
    )
    .map_err(py_value_error)?;
    // `plan_latent_dim` (computed above) is the optimizer's per-atom latent
    // dimension — `plan.latent_dim`, not the user-supplied `atom_dim` (periodic
    // atoms carry a harmonic count there; their `latent_dim == 1`).
    let effective_atom_dim: Vec<usize> = plan_latent_dim.clone();
    // Thread the per-atom Duchon centers into the inner driver so its
    // `DuchonCoordinateEvaluator` can re-evaluate `Phi(t)` / `dPhi/dt` at each
    // updated latent coordinate instead of freezing the seed snapshot.
    let atom_centers: Vec<Option<Array2<f64>>> = plans
        .iter()
        .map(|plan| plan.duchon_centers.clone())
        .collect();
    let fisher_u = fisher_factors.as_ref().map(|f| f.as_array());
    let fisher_mr = fisher_mass_residual.as_ref().map(|m| m.as_array());
    let row_w = row_loss_weights.as_ref().map(|w| w.as_array());
    let result_dict = sae_manifold_fit_inner(
        py,
        z_view,
        &atom_basis,
        effective_atom_dim,
        &atom_centers,
        basis_values.view(),
        basis_jacobian.view(),
        basis_sizes.clone(),
        decoder_coefficients.view(),
        smooth_penalties.view(),
        initial_logits.view(),
        start_coords.view(),
        alpha,
        tau,
        learnable_alpha,
        assignment_kind,
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        gumbel_schedule,
        analytic_penalties,
        top_k,
        jumprelu_threshold,
        native_ard_enabled,
        // Refine the cold routing seed (alternating coordinate projection +
        // weighted decoder LSQ) only when BOTH the logits and the coordinates
        // are cold — i.e. the auto seed is in control. A user-supplied warm
        // start (amortized encoder, #357) is respected verbatim.
        logits_are_cold && initial_coords.is_none(),
        random_state,
        // WP-D → fit wiring (#980): thread the optional output-Fisher shard
        // through so the auto facade installs `RowMetric::OutputFisher` the same
        // magic-by-default way as the precomputed-basis `sae_manifold_fit`.
        // Absent ⇒ the bit-identical Euclidean path.
        fisher_u,
        fisher_mr,
        fisher_provenance.as_deref(),
        row_w,
        separation_barrier_strength_override,
        ibp_alpha_override,
        structured_residual_passes,
        promote_from_residual,
        run_structure_search,
        run_outer_rho_search,
        quotient_scale,
        data_row_reseed,
        rank_charge_evidence,
    )?;
    // #977 — the per-atom `atom_plans` are now emitted by `sae_manifold_fit_inner`
    // FROM THE POST-SEARCH dictionary (variable K), so OOS predict can rebuild the
    // design for the DISCOVERED atoms (births included) without Python. The old
    // post-hoc attachment here re-derived plans from the seed `plans` (input K),
    // which would shadow the grown dictionary with a too-short list and break the
    // `from_payload` zip the moment a birth landed — exactly the plumbing
    // constraint #997 cited. The seed `plans` still build the cold-start design
    // and the per-atom Duchon centers threaded into the inner driver above; they
    // are no longer the serialized plan surface.
    Ok(result_dict)
}

/// Out-of-sample inference: same Newton driver as the fit path, with the
/// trained decoder blocks held frozen across iterations. `decoder_blocks` is
/// a per-atom list of `(M_k, p)` arrays; `duchon_centers` is `Some` only for
/// non-periodic atoms; `n_harmonics_list` is `Some` only for periodic atoms.
///
/// Returns the same full payload dict as the fit path (issue #357): the
/// converged per-token assignments `assignments_z` (N, K), per-atom
/// on-manifold coordinates `on_atom_coords_t`, gating logits, and the
/// reconstruction `fitted`. Downstream supervised heads consume the OOS
/// assignments directly, and the amortized-encoder loop reads the converged
/// coordinates as distillation targets. `initial_logits` (N, K) and
/// `initial_coords` (K, N, D_max) optionally warm-start the OOS refinement
/// from an encoder's per-token prediction.
// Convert borrowed FFI arrays into the owned typed library request. This helper
// performs wire parsing and ownership transfer only; validation, basis rebuild,
// seeding, inference, projection, and reporting live in gam-sae.
fn sae_oos_request_from_arrays(
    x_view: ndarray::ArrayView2<'_, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
    duchon_centers: &[Option<Array2<f64>>],
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    alpha: f64,
    tau: f64,
    assignment_kind: String,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    initial_logits: Option<ndarray::ArrayView2<'_, f64>>,
    initial_coords: Option<ndarray::ArrayView3<'_, f64>>,
    jumprelu_threshold: f64,
    top_k: Option<usize>,
    hybrid_linear_images: Option<Vec<(usize, f64, Array1<f64>, Array1<f64>, Option<Array1<f64>>)>>,
    log_lambda_sparse: Option<f64>,
    log_lambda_smooth: Option<Vec<f64>>,
    log_ard: Option<Vec<Vec<f64>>>,
    learnable_alpha: bool,
) -> Result<gam::terms::sae::manifold::SaeOosRequest, String> {
    let k_atoms = atom_basis.len();
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
    {
        return Err(format!(
            "sae_manifold_predict_oos: per-atom metadata lengths must equal K={k_atoms}"
        ));
    }
    let assignment = match assignment_kind.as_str() {
        "softmax" => gam::terms::sae::manifold::SaeOosAssignmentKind::Softmax,
        "ibp_map" => gam::terms::sae::manifold::SaeOosAssignmentKind::IbpMap { learnable_alpha },
        "threshold_gate" => gam::terms::sae::manifold::SaeOosAssignmentKind::ThresholdGate {
            threshold: jumprelu_threshold,
        },
        "topk" => gam::terms::sae::manifold::SaeOosAssignmentKind::TopK,
        _ => {
            return Err(format!(
                "sae_manifold_predict_oos: unsupported assignment kind {assignment_kind:?}"
            ));
        }
    };
    let regularization = match (log_lambda_sparse, log_lambda_smooth, log_ard) {
        (Some(log_lambda_sparse), Some(log_lambda_smooth), Some(log_ard)) => {
            gam::terms::sae::manifold::SaeOosRegularization {
                log_lambda_sparse,
                log_lambda_smooth,
                log_ard,
            }
        }
        _ => {
            return Err(
                "sae_manifold_predict_oos: terminal rho must provide log_lambda_sparse, \
                 log_lambda_smooth, and log_ard together"
                    .to_string(),
            );
        }
    };
    let atoms = (0..k_atoms)
        .map(|atom_index| gam::terms::sae::manifold::SaeOosAtomSpec {
            basis_kind: sae_atom_basis_kind_from_str(&atom_basis[atom_index]),
            latent_dim: atom_dim[atom_index],
            decoder: decoder_blocks[atom_index].to_owned(),
            centers: duchon_centers[atom_index].clone(),
            n_harmonics: n_harmonics_list[atom_index],
            basis_size: basis_size_list[atom_index],
        })
        .collect();
    let hybrid_linear_images = match hybrid_linear_images {
        Some(images) => images,
        None => Vec::new(),
    }
    .into_iter()
    .map(
        |(atom_idx, t_bar, b0, b1, v)| gam::terms::sae::hybrid_split::AtomLinearImage {
            atom_idx,
            t_bar,
            b0,
            b1,
            row_codes: None,
            v,
        },
    )
    .collect();

    Ok(gam::terms::sae::manifold::SaeOosRequest {
        target: x_view.to_owned(),
        atoms,
        assignment,
        alpha,
        tau,
        regularization,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        initial_logits: initial_logits.map(|view| view.to_owned()),
        initial_coords: initial_coords.map(|view| view.to_owned()),
        top_k,
        hybrid_linear_images,
    })
}

// Serialize the typed library report. No fit or inference decisions live here.
fn sae_oos_report_to_pydict<'py>(
    py: Python<'py>,
    report: gam::terms::sae::manifold::SaeOosReport,
) -> PyResult<Py<PyDict>> {
    let chosen_k = report.active_mask.len();
    let atoms_py = PyList::empty(py);
    for atom in report.atoms {
        let atom_dict = PyDict::new(py);
        atom_dict.set_item("decoder_B", atom.decoder.into_pyarray(py))?;
        atom_dict.set_item("basis_kind", sae_atom_basis_kind_name(&atom.basis_kind))?;
        atom_dict.set_item("basis_centers", py.None())?;
        atom_dict.set_item("on_atom_coords_t", atom.coords.into_pyarray(py))?;
        atom_dict.set_item("assignments_z", atom.assignments.into_pyarray(py))?;
        atom_dict.set_item("active_dim", atom.active_dim)?;
        atom_dict.set_item("atom_reconstruction", atom.reconstruction.into_pyarray(py))?;
        atoms_py.append(atom_dict)?;
    }
    let log_ard_py = PyList::empty(py);
    for atom_log_ard in &report.rho.log_ard {
        log_ard_py.append(atom_log_ard.clone().into_pyarray(py))?;
    }

    let out = PyDict::new(py);
    out.set_item("atoms", atoms_py)?;
    out.set_item("assignments_z", report.assignments.into_pyarray(py))?;
    out.set_item("logits", report.logits.into_pyarray(py))?;
    out.set_item("atom_active_mask", report.active_mask)?;
    out.set_item("fitted", report.fitted.into_pyarray(py))?;
    sae_set_penalized_loss_items(&out, &report.loss, "oos_penalized_loss")?;
    out.set_item("log_alpha", report.alpha.ln())?;
    out.set_item("log_lambda_smooth", report.rho.log_lambda_smooth)?;
    out.set_item("log_ard", log_ard_py)?;
    out.set_item("assignment_prior", report.assignment_kind)?;
    out.set_item(
        "solver_plan",
        sae_streaming_plan_to_pydict(py, report.streaming_plan)?,
    )?;
    out.set_item("chosen_k", chosen_k)?;
    Ok(out.unbind())
}

/// FFI surface for the frozen-decoder out-of-sample solve. The binding only
/// marshals arrays into [`SaeOosRequest`](gam::terms::sae::manifold::SaeOosRequest),
/// calls the typed gam-sae entry, and serializes its report (#2236).
#[pyfunction(signature = (
    x_new,
    atom_basis,
    atom_dim,
    decoder_blocks,
    duchon_centers,
    n_harmonics_list,
    basis_size_list,
    alpha,
    tau,
    assignment_kind,
    max_iter = 50,
    learning_rate = 0.04,
    ridge_ext_coord = 1.0e-6,
    initial_logits = None,
    initial_coords = None,
    jumprelu_threshold = 0.0,
    top_k = None,
    hybrid_linear_images = None,
    log_lambda_sparse = None,
    log_lambda_smooth = None,
    log_ard = None,
    learnable_alpha = false,
))]
fn sae_manifold_predict_oos<'py>(
    py: Python<'py>,
    x_new: PyReadonlyArray2<'py, f64>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    duchon_centers: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    alpha: f64,
    tau: f64,
    assignment_kind: String,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    initial_logits: Option<PyReadonlyArray2<'py, f64>>,
    initial_coords: Option<PyReadonlyArray3<'py, f64>>,
    jumprelu_threshold: f64,
    top_k: Option<usize>,
    hybrid_linear_images: Option<
        Vec<(
            usize,
            f64,
            PyReadonlyArray1<'py, f64>,
            PyReadonlyArray1<'py, f64>,
            Option<PyReadonlyArray1<'py, f64>>,
        )>,
    >,
    log_lambda_sparse: Option<f64>,
    log_lambda_smooth: Option<Vec<f64>>,
    log_ard: Option<Vec<Vec<f64>>>,
    learnable_alpha: bool,
) -> PyResult<Py<PyDict>> {
    let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
        decoder_blocks.iter().map(|b| b.as_array()).collect();
    let duchon_owned: Vec<Option<Array2<f64>>> = duchon_centers
        .iter()
        .map(|o| o.as_ref().map(|a| a.as_array().to_owned()))
        .collect();
    let initial_logits_view = initial_logits.as_ref().map(|a| a.as_array());
    let initial_coords_view = initial_coords.as_ref().map(|a| a.as_array());
    let hybrid_owned = hybrid_linear_images.map(|images| {
        images
            .into_iter()
            .map(|(atom_idx, t_bar, b0, b1, v)| {
                (
                    atom_idx,
                    t_bar,
                    b0.as_array().to_owned(),
                    b1.as_array().to_owned(),
                    v.map(|arr| arr.as_array().to_owned()),
                )
            })
            .collect()
    });
    let request = sae_oos_request_from_arrays(
        x_new.as_array(),
        atom_basis,
        atom_dim,
        &decoder_views,
        &duchon_owned,
        n_harmonics_list,
        basis_size_list,
        alpha,
        tau,
        assignment_kind,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        initial_logits_view,
        initial_coords_view,
        jumprelu_threshold,
        top_k,
        hybrid_owned,
        log_lambda_sparse,
        log_lambda_smooth,
        log_ard,
        learnable_alpha,
    )
    .map_err(py_value_error)?;
    let report =
        gam::terms::sae::manifold::run_sae_manifold_oos(request).map_err(py_value_error)?;
    sae_oos_report_to_pydict(py, report)
}

/// (#1010) A frozen-dictionary Kantorovich-certified encode atlas, exposed to
/// Python. Built once over a fitted SAE dictionary; [`Self::certified_encode`]
/// then runs the per-atom certified batch and returns the per-row `h ≤ ½`
/// Newton–Kantorovich certificate flag — the honesty signal an amortized encoder
/// reads INSTEAD of a cold exact multi-start probe per row
/// (`ManifoldSAE._oos_payload`). Uncertified rows are flagged so the caller still
/// routes them to the exact fallback; no approximation enters silently.
#[pyclass(name = "SaeEncodeAtlas", unsendable)]
pub struct PySaeEncodeAtlas {
    atlas: gam::terms::sae::encode::EncodeAtlas,
    atoms: Vec<gam::terms::sae::manifold::SaeManifoldAtom>,
    latent_dims: Vec<usize>,
}

#[pymethods]
impl PySaeEncodeAtlas {
    /// Number of atoms the atlas covers.
    #[getter]
    fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Per-atom latent dimensionalities.
    #[getter]
    fn latent_dims(&self) -> Vec<usize> {
        self.latent_dims.clone()
    }

    /// Certified encode of `x` `(N, p)` against atom `atom_index`, at the fixed
    /// per-row `amplitudes` `(N,)`. Returns `{coords (N, d), certified (N,) bool,
    /// n_uncertified, latent_dim}`. `certified[i]` is the row's `h ≤ ½`
    /// certificate; a `False` row must be routed to the exact fallback
    /// (`ManifoldSAE.converged_latents`) — the honesty gate.
    #[pyo3(signature = (x, amplitudes, atom_index))]
    fn certified_encode<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        amplitudes: PyReadonlyArray1<'py, f64>,
        atom_index: usize,
    ) -> PyResult<Py<PyDict>> {
        if atom_index >= self.atoms.len() {
            return Err(py_value_error(format!(
                "SaeEncodeAtlas.certified_encode: atom_index {atom_index} out of range (K={})",
                self.atoms.len()
            )));
        }
        let targets = x.as_array();
        let amps = amplitudes.as_array();
        if amps.len() != targets.nrows() {
            return Err(py_value_error(format!(
                "SaeEncodeAtlas.certified_encode: amplitudes length {} != rows {}",
                amps.len(),
                targets.nrows()
            )));
        }
        let result = self
            .atlas
            .certified_encode_batch(&self.atoms[atom_index], atom_index, targets, amps)
            .map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("coords", result.coords.into_pyarray(py))?;
        out.set_item("certified", result.certified)?;
        out.set_item("n_uncertified", result.encode_uncertified_count)?;
        out.set_item("latent_dim", self.latent_dims[atom_index])?;
        Ok(out.unbind())
    }

    /// Decode recovered latent `coords` `(N, d)` back through atom `atom_index`'s
    /// frozen basis + decoder, scaled per row by `amplitudes`: `x̂ = z·Φ(t)·B`.
    /// The ambient-space companion to [`Self::certified_encode`], so a caller can
    /// compare a certified encode's reconstruction against the exact fitted
    /// reconstruction without touching the coordinate gauge.
    #[pyo3(signature = (coords, amplitudes, atom_index))]
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArray2<'py, f64>,
        amplitudes: PyReadonlyArray1<'py, f64>,
        atom_index: usize,
    ) -> PyResult<Py<PyArray2<f64>>> {
        if atom_index >= self.atoms.len() {
            return Err(py_value_error(format!(
                "SaeEncodeAtlas.reconstruct: atom_index {atom_index} out of range (K={})",
                self.atoms.len()
            )));
        }
        let atom = &self.atoms[atom_index];
        let coords_view = coords.as_array();
        let amps = amplitudes.as_array();
        if coords_view.nrows() != amps.len() {
            return Err(py_value_error(format!(
                "SaeEncodeAtlas.reconstruct: coords rows {} != amplitudes length {}",
                coords_view.nrows(),
                amps.len()
            )));
        }
        let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
            py_value_error("SaeEncodeAtlas.reconstruct: atom has no evaluator".into())
        })?;
        let (phi, _jet) = evaluator.evaluate(coords_view).map_err(py_value_error)?;
        let mut recon = phi.dot(&atom.decoder_coefficients);
        for i in 0..recon.nrows() {
            let z = amps[i];
            let mut row = recon.row_mut(i);
            row.map_inplace(|v| *v *= z);
        }
        Ok(recon.into_pyarray(py).unbind())
    }
}

/// The DISTILLED / AMORTIZED encoder, exposed to Python (reviewer condition #2).
///
/// Our held-out reconstruction comes from a per-row test-time optimization; a
/// sparse-autoencoder's comes from one matmul. This encoder is the one-matmul
/// distilled map: fit against a fit's EXACT per-row code (gate logits, per-atom
/// coords, amplitudes) by closed-form evidence maximization, it predicts that
/// code for fresh rows in a single matmul. `encode_amortized(X)` is the PRIMARY
/// deployable out-of-sample encode; the exact `sae_manifold_predict_oos` solve is
/// the ORACLE line, and the difference is the amortization gap.
#[pyclass(name = "SaeAmortizedEncoder", unsendable)]
pub struct PySaeAmortizedEncoder {
    encoder: gam::terms::sae::amortized_encoder::LearnedAmortizedEncoder,
}

#[pymethods]
impl PySaeAmortizedEncoder {
    /// Fit the encoder against a fit's exact per-row code. `train_x` is the
    /// `(n, p)` training corpus; `train_logits` and `train_amplitudes` are
    /// `(n, K)`; `train_coords` is one `(n, d_k)` block per atom (the exact
    /// solver's converged coordinates). The evidence chooses the encoder's
    /// capacity (linear vs a diagonal-quadratic head).
    #[new]
    #[pyo3(signature = (train_x, train_logits, train_coords, train_amplitudes, coord_periods))]
    fn new<'py>(
        train_x: PyReadonlyArray2<'py, f64>,
        train_logits: PyReadonlyArray2<'py, f64>,
        train_coords: Vec<PyReadonlyArray2<'py, f64>>,
        train_amplitudes: PyReadonlyArray2<'py, f64>,
        coord_periods: Vec<Vec<Option<f64>>>,
    ) -> PyResult<Self> {
        let coords: Vec<Array2<f64>> = train_coords
            .iter()
            .map(|c| c.as_array().to_owned())
            .collect();
        let encoder = gam::terms::sae::amortized_encoder::LearnedAmortizedEncoder::fit_with_axis_periods(
            train_x.as_array(),
            train_logits.as_array(),
            &coords,
            train_amplitudes.as_array(),
            &coord_periods,
        )
        .map_err(py_value_error)?;
        Ok(Self { encoder })
    }

    /// Pooled log marginal likelihood of the trained encoder (the evidence).
    #[getter]
    fn log_evidence(&self) -> f64 {
        self.encoder.log_evidence
    }

    /// Whether the evidence admitted the diagonal-quadratic head over the linear
    /// null (capacity justified by evidence, not a knob).
    #[getter]
    fn used_quadratic_head(&self) -> bool {
        self.encoder.used_quadratic_head
    }

    /// Number of features in the winning design.
    #[getter]
    fn feature_dim(&self) -> usize {
        self.encoder.feature_dim
    }

    /// Effective degrees of freedom per target of the trained encoder.
    #[getter]
    fn effective_dof(&self) -> f64 {
        self.encoder.effective_dof
    }

    /// Number of atoms the encoder predicts a code for.
    #[getter]
    fn k_atoms(&self) -> usize {
        self.encoder.k_atoms()
    }

    /// One-matmul encode of fresh rows `x` `(m, p)`. Returns `{logits (m, K),
    /// coords: list of (m, d_k), amplitudes (m, K)}` — the distilled per-row code
    /// in the exact solver's layout. Amplitudes are clamped at zero (masses are
    /// non-negative).
    #[pyo3(signature = (x))]
    fn encode_amortized<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let code = self.encoder.predict(x.as_array()).map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item("logits", code.logits.into_pyarray(py))?;
        let coords_py = PyList::empty(py);
        for c in code.coords {
            coords_py.append(c.into_pyarray(py))?;
        }
        out.set_item("coords", coords_py)?;
        out.set_item("amplitudes", code.amplitudes.into_pyarray(py))?;
        Ok(out.unbind())
    }
}

/// (#1010) Build a Kantorovich-certified [`PySaeEncodeAtlas`] over a fitted SAE
/// dictionary. `decoder_blocks[k]` is atom `k`'s frozen `(M_k, p)` decoder;
/// `amplitude_bounds[k]` bounds `|z_k|` and `target_norm_bound` bounds `‖x‖` over
/// the encode data. Both scale the offline Hessian-Lipschitz constant `L`, so a
/// larger bound only SHRINKS the certified radius — it can never issue a false
/// certificate. Precomputed bases (no analytic second jet) are rejected: they
/// have no closed-form `L` and must route to the exact encode.
///
/// When `amplitude_bounds` is `None` the per-atom default `|z_k|` bound is the
/// max `|assignment|` over the `(N, K)` training `assignments` (or `1.0` for an
/// empty design); when `target_norm_bound` is `None` the default `‖x‖` bound is
/// the max row `L2` norm of the `(N, F)` `encode_rows` (or `1.0` when empty).
/// These default reductions live here so the wrapper hands the arrays over
/// verbatim instead of pre-reducing them in NumPy.
#[pyfunction(signature = (
    basis_kinds,
    atom_dims,
    decoder_blocks,
    duchon_centers,
    basis_sizes,
    amplitude_bounds,
    target_norm_bound,
    assignments = None,
    encode_rows = None,
    grid_resolution = 32,
    ridge = 1.0e-10,
    newton_steps = 8,
))]
fn build_sae_encode_atlas<'py>(
    basis_kinds: Vec<String>,
    atom_dims: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    duchon_centers: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    basis_sizes: Vec<usize>,
    amplitude_bounds: Option<Vec<f64>>,
    target_norm_bound: Option<f64>,
    assignments: Option<PyReadonlyArray2<'py, f64>>,
    encode_rows: Option<PyReadonlyArray2<'py, f64>>,
    grid_resolution: usize,
    ridge: f64,
    newton_steps: usize,
) -> PyResult<PySaeEncodeAtlas> {
    let k_atoms = basis_kinds.len();
    if k_atoms == 0 {
        return Err(py_value_error(
            "build_sae_encode_atlas: dictionary must have at least one atom".into(),
        ));
    }
    // Default amplitude bounds: per-atom max |assignment| over the training
    // codes (or 1.0 for an empty design), matching the former NumPy reduction.
    let amplitude_bounds: Vec<f64> = match amplitude_bounds {
        Some(bounds) => bounds,
        None => {
            let assignments = assignments.ok_or_else(|| {
                py_value_error(
                    "build_sae_encode_atlas: amplitude_bounds=None requires the training \
                     assignments array"
                        .into(),
                )
            })?;
            let a = assignments.as_array();
            if a.ncols() < k_atoms {
                return Err(py_value_error(format!(
                    "build_sae_encode_atlas: assignments has {} columns but K={k_atoms}",
                    a.ncols()
                )));
            }
            (0..k_atoms)
                .map(|k| {
                    if a.nrows() == 0 {
                        1.0
                    } else {
                        a.column(k)
                            .iter()
                            .map(|v| v.abs())
                            .fold(f64::NEG_INFINITY, f64::max)
                    }
                })
                .collect()
        }
    };
    // Default target-norm bound: max row L2 norm of the encode data (or 1.0 for
    // an empty matrix), matching the former NumPy reduction.
    let target_norm_bound: f64 = match target_norm_bound {
        Some(bound) => bound,
        None => {
            let rows = encode_rows.ok_or_else(|| {
                py_value_error(
                    "build_sae_encode_atlas: target_norm_bound=None requires the encode-data rows"
                        .into(),
                )
            })?;
            let x = rows.as_array();
            if x.len() == 0 {
                1.0
            } else {
                x.rows()
                    .into_iter()
                    .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt())
                    .fold(f64::NEG_INFINITY, f64::max)
            }
        }
    };
    if atom_dims.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || basis_sizes.len() != k_atoms
        || amplitude_bounds.len() != k_atoms
    {
        return Err(py_value_error(format!(
            "build_sae_encode_atlas: per-atom metadata lengths must all equal K={k_atoms}"
        )));
    }
    let kinds: Vec<SaeAtomBasisKind> = basis_kinds
        .iter()
        .map(|s| sae_atom_basis_kind_from_str(s))
        .collect();
    let centers: Vec<Option<Array2<f64>>> = duchon_centers
        .iter()
        .map(|c| c.as_ref().map(|arr| arr.as_array().to_owned()))
        .collect();
    // Per-atom seed coordinate (origin): the atom's stored basis values are only
    // placeholders — the certified encode re-evaluates Φ(t) through the live
    // evaluator — so a single valid coordinate satisfies `SaeManifoldAtom::new`.
    let coord_blocks: Vec<Array2<f64>> = atom_dims
        .iter()
        .map(|&d| Array2::<f64>::zeros((1, d.max(1))))
        .collect();
    let evaluators =
        build_sae_basis_evaluators(&kinds, &basis_sizes, &atom_dims, &coord_blocks, &centers)
            .map_err(py_value_error)?;
    let mut atoms: Vec<gam::terms::sae::manifold::SaeManifoldAtom> = Vec::with_capacity(k_atoms);
    let mut latent_dims: Vec<usize> = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let d = atom_dims[k];
        let evaluator = evaluators[k].clone().ok_or_else(|| {
            py_value_error(format!(
                "build_sae_encode_atlas: atom {k} basis has no analytic second-jet evaluator; \
                 cannot certify (precomputed bases route to the exact encode)"
            ))
        })?;
        let (seed_phi, seed_jet) = evaluator
            .evaluate(coord_blocks[k].view())
            .map_err(py_value_error)?;
        let m = seed_phi.ncols();
        let decoder = decoder_blocks[k].as_array().to_owned();
        if decoder.nrows() != m {
            return Err(py_value_error(format!(
                "build_sae_encode_atlas: decoder_blocks[{k}] has M={} but the rebuilt basis has \
                 M={m}; basis_kinds / atom_dims / basis_sizes / duchon_centers must match the \
                 trained design",
                decoder.nrows()
            )));
        }
        let atom = gam::terms::sae::manifold::SaeManifoldAtom::new(
            format!("atom{k}"),
            kinds[k].clone(),
            d,
            seed_phi,
            seed_jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .map_err(py_value_error)?
        .with_basis_second_jet(evaluator);
        atoms.push(atom);
        latent_dims.push(d);
    }
    let config = gam::terms::sae::encode::AtlasConfig {
        grid_resolution,
        ridge,
        newton_steps,
    };
    let atlas = gam::terms::sae::encode::EncodeAtlas::build(
        &atoms,
        &amplitude_bounds,
        target_norm_bound,
        config,
    )
    .map_err(py_value_error)?;
    Ok(PySaeEncodeAtlas {
        atlas,
        atoms,
        latent_dims,
    })
}

/// Compute a steering plan with output dosimetry for a fitted SAE-manifold atom
/// ([`gam::inference::steering::steer_delta`]).
///
/// This is the FFI surface for the steering primitive: it rebuilds the fitted
/// [`gam::terms::sae::manifold::SaeManifoldTerm`] from the trained decoder blocks
/// + basis metadata, seeds it with the *trained* on-atom coordinates and routing
/// logits (no re-solve — the model is fixed), optionally installs the WP-D
/// per-row output-Fisher metric ([`gam::inference::row_metric::RowMetric::output_fisher`])
/// from `fisher_factors` (the same shard the fit used), and calls `steer_delta`
/// to drive atom `atom_k` from `t_from` to `t_to`. It returns the
/// [`gam::inference::steering::SteerPlan`] fields as a dict: the activation-space
/// `delta`, the path-integrated `predicted_nats` dose, the `validity_radius`,
/// the `off_manifold_norm` self-check, and the `metric_provenance`.
///
/// The term rebuild mirrors [`sae_manifold_predict_oos`] (same plan/evaluator
/// machinery), but where `predict_oos` runs the frozen-decoder Newton solve on a
/// *new* `X`, this seeds the term directly from the trained latents/logits so the
/// dose is measured through the model as fitted. `coords` is one `(N, d_k)` array
/// per atom (the trained `on_atom_coords_t`); `logits` is `(N, K)` (the trained
/// routing logits) so the per-atom amplitude / measured-row selection inside
/// `steer_delta` sees the fitted assignments. `fisher_factors` is the `(n, p, r)`
/// harvest shard `U`; its presence installs `RowMetric::OutputFisher` (and makes
/// `predicted_nats` / `validity_radius` available), exactly as in the fit.
/// Owned-array core of the steering primitive (#2091): the full per-atom basis
/// rebuild + trained-latent seeding + optional output-Fisher metric install +
/// `steer_delta` call, on borrowed ndarray views instead of `PyReadonlyArray`.
///
/// Both callers route through this single rebuild path so their `SteerPlan`s are
/// identical by construction: the `sae_steer_delta` `#[pyfunction]` (arrays
/// marshalled from Python) and `ManifoldSaeCore::steer` (arrays read from the
/// Rust-owned model state, so an attached Fisher shard is NOT re-marshalled
/// across the FFI boundary per call — acceptance bullet 2). The
/// predicted-nats-vs-analytic steering tests guard this rebuild's correctness;
/// the pyclass equivalence test guards that the two callers thread identical
/// inputs into it. `fisher_provenance`: same-position `"output_fisher"` (default)
/// or forward-looking `"output_fisher_downstream"` — selects the re-installed
/// output-Fisher `RowMetric` the dose is measured through.
fn steer_delta_from_arrays(
    atom_k: usize,
    t_from: ndarray::ArrayView1<'_, f64>,
    t_to: ndarray::ArrayView1<'_, f64>,
    n_obs: usize,
    p_out: usize,
    atom_basis: &[String],
    atom_dim: &[usize],
    decoder_blocks: &[ndarray::ArrayView2<'_, f64>],
    duchon_centers: &[Option<Array2<f64>>],
    n_harmonics_list: &[Option<usize>],
    basis_size_list: &[usize],
    coords: &[ndarray::ArrayView2<'_, f64>],
    logits: ndarray::ArrayView2<'_, f64>,
    assignment_kind: &str,
    tau: f64,
    alpha: f64,
    jumprelu_threshold: f64,
    fisher_factors: Option<ndarray::ArrayView3<'_, f64>>,
    fisher_provenance: Option<&str>,
) -> PyResult<gam::inference::steering::SteerPlan> {
    // #1777 — accept both "threshold_gate" (primary) and legacy "jumprelu".
    let assignment_kind = canonicalize_assignment_kind(assignment_kind).map_err(py_value_error)?;
    let k_atoms = atom_basis.len();
    // Guard the per-atom metadata lengths before indexing them into the atom
    // specs below; every other precondition (positive dims, atom_k range, coord /
    // logit shapes, positive alpha/tau) is validated inside the engine entry.
    if atom_dim.len() != k_atoms
        || decoder_blocks.len() != k_atoms
        || duchon_centers.len() != k_atoms
        || n_harmonics_list.len() != k_atoms
        || basis_size_list.len() != k_atoms
        || coords.len() != k_atoms
    {
        return Err(py_value_error(format!(
            "sae_steer_delta: per-atom metadata lengths must equal K={k_atoms}"
        )));
    }
    let assignment = match assignment_kind.as_str() {
        "softmax" => gam::terms::sae::manifold::SaeOosAssignmentKind::Softmax,
        "ibp_map" => gam::terms::sae::manifold::SaeOosAssignmentKind::IbpMap {
            learnable_alpha: false,
        },
        "threshold_gate" => gam::terms::sae::manifold::SaeOosAssignmentKind::ThresholdGate {
            threshold: jumprelu_threshold,
        },
        "topk" => gam::terms::sae::manifold::SaeOosAssignmentKind::TopK,
        _ => {
            return Err(py_value_error(format!(
                "sae_steer_delta: assignment_kind must be one of 'softmax', 'ibp_map', 'threshold_gate', or 'topk' (legacy alias 'jumprelu' also accepted); got {assignment_kind}"
            )));
        }
    };

    // Marshal the persisted dictionary schema into typed OOS atom specs — the SAME
    // rebuild contract `sae_manifold_predict_oos` marshals into, so the steer term
    // and the OOS term are rebuilt by one engine path (`run_sae_manifold_steer`,
    // #2236) rather than a duplicated pyffi rebuild.
    let atoms: Vec<gam::terms::sae::manifold::SaeOosAtomSpec> = (0..k_atoms)
        .map(|atom_index| gam::terms::sae::manifold::SaeOosAtomSpec {
            basis_kind: sae_atom_basis_kind_from_str(&atom_basis[atom_index]),
            latent_dim: atom_dim[atom_index],
            decoder: decoder_blocks[atom_index].to_owned(),
            centers: duchon_centers[atom_index].clone(),
            n_harmonics: n_harmonics_list[atom_index],
            basis_size: basis_size_list[atom_index],
        })
        .collect();
    let coord_blocks: Vec<Array2<f64>> = coords.iter().map(|block| block.to_owned()).collect();

    // Marshal the WP-D output-Fisher shard into a `RowMetric` (array plumbing only):
    // the `(n, p, r)` → `(n, p*r)` flatten the fit uses plus the provenance
    // selection. Its presence installs the OutputFisher metric inside the engine so
    // `predicted_nats` / `validity_radius` are available; absence keeps the
    // geometry-only Euclidean path (dose degrades to None).
    let fisher_metric = match fisher_factors {
        Some(u3) => {
            let u_shape = u3.shape();
            if u_shape[0] != n_obs || u_shape[1] != p_out {
                return Err(py_value_error(format!(
                    "sae_steer_delta: fisher_factors U must be (n, p, r)=({n_obs}, {p_out}, r); \
                     got leading dims ({}, {})",
                    u_shape[0], u_shape[1]
                )));
            }
            let rank = u_shape[2];
            if rank == 0 {
                return Err(py_value_error(
                    "sae_steer_delta: fisher_factors U rank (last axis) must be >= 1".to_string(),
                ));
            }
            if rank > p_out {
                return Err(py_value_error(format!(
                    "sae_steer_delta: fisher_factors U rank {rank} exceeds output dim p={p_out}"
                )));
            }
            if !u3.iter().all(|v| v.is_finite()) {
                return Err(py_value_error(
                    "sae_steer_delta: fisher_factors U contains non-finite values".into(),
                ));
            }
            let mut u_flat = Array2::<f64>::zeros((n_obs, p_out * rank));
            for row in 0..n_obs {
                for i in 0..p_out {
                    for k in 0..rank {
                        u_flat[[row, i * rank + k]] = u3[[row, i, k]];
                    }
                }
            }
            Some(row_metric_from_fisher_provenance(
                u_flat,
                p_out,
                rank,
                fisher_provenance,
            )?)
        }
        None => None,
    };

    let request = gam::terms::sae::manifold::SaeSteerRequest {
        atoms,
        coords: coord_blocks,
        logits: logits.to_owned(),
        assignment,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        t_from: t_from.to_vec(),
        t_to: t_to.to_vec(),
    };
    gam::terms::sae::manifold::run_sae_manifold_steer(request).map_err(py_value_error)
}

/// Render a [`gam::inference::steering::SteerPlan`] as the Python dict both steer
/// callers return (the `sae_steer_delta` pyfunction and `ManifoldSaeCore::steer`).
fn steer_plan_to_pydict(
    py: Python<'_>,
    plan: gam::inference::steering::SteerPlan,
) -> PyResult<Py<PyDict>> {
    let provenance_str = metric_provenance_label(plan.metric_provenance);
    let out = PyDict::new(py);
    out.set_item("atom", plan.atom)?;
    out.set_item("atom_name", plan.atom_name)?;
    out.set_item("t_from", plan.t_from)?;
    out.set_item("t_to", plan.t_to)?;
    out.set_item("amplitude", plan.amplitude)?;
    out.set_item("measured_row", plan.measured_row)?;
    out.set_item("delta", plan.delta.into_pyarray(py))?;
    out.set_item("predicted_nats", plan.predicted_nats)?;
    out.set_item("validity_radius", plan.validity_radius)?;
    out.set_item("off_manifold_norm", plan.off_manifold_norm)?;
    out.set_item("metric_provenance", provenance_str)?;
    Ok(out.unbind())
}

/// FFI surface for the steering primitive: rebuilds the fitted term from the
/// trained decoder blocks + basis metadata, seeds it with the trained latents /
/// logits (no re-solve), optionally installs the WP-D output-Fisher metric from
/// `fisher_factors`, and drives atom `atom_k` from `t_from` to `t_to`, returning
/// the [`gam::inference::steering::SteerPlan`] as a dict. Thin marshalling over
/// the shared [`steer_delta_from_arrays`] rebuild (#2091).
#[pyfunction(signature = (
    atom_k,
    t_from,
    t_to,
    n_obs,
    p_out,
    atom_basis,
    atom_dim,
    decoder_blocks,
    duchon_centers,
    n_harmonics_list,
    basis_size_list,
    coords,
    logits,
    assignment_kind,
    tau,
    alpha = 1.0,
    jumprelu_threshold = 0.0,
    fisher_factors = None,
    fisher_provenance = None,
))]
fn sae_steer_delta<'py>(
    py: Python<'py>,
    atom_k: usize,
    t_from: PyReadonlyArray1<'py, f64>,
    t_to: PyReadonlyArray1<'py, f64>,
    n_obs: usize,
    p_out: usize,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    duchon_centers: Vec<Option<PyReadonlyArray2<'py, f64>>>,
    n_harmonics_list: Vec<Option<usize>>,
    basis_size_list: Vec<usize>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    logits: PyReadonlyArray2<'py, f64>,
    assignment_kind: String,
    tau: f64,
    alpha: f64,
    jumprelu_threshold: f64,
    fisher_factors: Option<PyReadonlyArray3<'py, f64>>,
    fisher_provenance: Option<String>,
) -> PyResult<Py<PyDict>> {
    let decoder_views: Vec<ndarray::ArrayView2<'_, f64>> =
        decoder_blocks.iter().map(|b| b.as_array()).collect();
    let coord_views: Vec<ndarray::ArrayView2<'_, f64>> =
        coords.iter().map(|c| c.as_array()).collect();
    let duchon_owned: Vec<Option<Array2<f64>>> = duchon_centers
        .iter()
        .map(|o| o.as_ref().map(|a| a.as_array().to_owned()))
        .collect();
    let fisher_view = fisher_factors.as_ref().map(|f| f.as_array());
    let plan = steer_delta_from_arrays(
        atom_k,
        t_from.as_array(),
        t_to.as_array(),
        n_obs,
        p_out,
        &atom_basis,
        &atom_dim,
        &decoder_views,
        &duchon_owned,
        &n_harmonics_list,
        &basis_size_list,
        &coord_views,
        logits.as_array(),
        &assignment_kind,
        tau,
        alpha,
        jumprelu_threshold,
        fisher_view,
        fisher_provenance.as_deref(),
    )?;
    steer_plan_to_pydict(py, plan)
}

/// Global coefficient of determination
/// R^2 = 1 - (Σ_ij (y_ij - ŷ_ij)²) / (Σ_ij (y_ij - mean_j)²)
/// for a fitted SAE-manifold reconstruction, where `mean_j` is the per-column
/// mean of the observed matrix. Both SSR and SST are summed across all rows and
/// columns, so this returns a single scalar (a global metric), not a vector of
/// per-column R² values. Pure-Rust closed-form so the Python wrapper is one FFI
/// call.
#[pyfunction]
fn sae_manifold_reconstruction_r2(
    observed: PyReadonlyArray2<'_, f64>,
    fitted: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let observed = observed.as_array();
    let fitted = fitted.as_array();
    if observed.dim() != fitted.dim() {
        return Err(py_value_error(format!(
            "sae_manifold_reconstruction_r2: shape mismatch observed={:?} fitted={:?}",
            observed.dim(),
            fitted.dim(),
        )));
    }
    let n_rows = observed.nrows();
    let n_cols = observed.ncols();
    if n_rows == 0 || n_cols == 0 {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: observed and fitted must be non-empty".into(),
        ));
    }
    if !observed.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: observed contains non-finite values".into(),
        ));
    }
    if !fitted.iter().all(|v| v.is_finite()) {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: fitted contains non-finite values".into(),
        ));
    }
    let mut col_means = vec![0.0_f64; n_cols];
    for col in 0..n_cols {
        let mut acc = 0.0;
        for row in 0..n_rows {
            acc += observed[[row, col]];
        }
        col_means[col] = acc / n_rows as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for row in 0..n_rows {
        for col in 0..n_cols {
            let d = observed[[row, col]] - fitted[[row, col]];
            ssr += d * d;
            let dm = observed[[row, col]] - col_means[col];
            sst += dm * dm;
        }
    }
    if !ssr.is_finite() || !sst.is_finite() {
        return Err(py_value_error(
            "sae_manifold_reconstruction_r2: SSR/SST overflowed; inputs have extreme magnitudes"
                .into(),
        ));
    }
    if sst == 0.0 {
        return Ok(f64::NAN);
    }
    Ok(1.0 - ssr / sst)
}

/// Sparsity summary stats for an `(n_rows, K)` assignment matrix returned by
/// `sae_manifold_fit*`. Returns `(avg_active_atoms, mean_assignment_mass)` where
/// "active" is `assignment >= threshold`.
#[pyfunction]
fn sae_manifold_assignment_summary(
    assignments: PyReadonlyArray2<'_, f64>,
    threshold: f64,
) -> PyResult<(f64, f64)> {
    if !threshold.is_finite() {
        return Err(py_value_error(
            "sae_manifold_assignment_summary: threshold must be finite".into(),
        ));
    }
    let a = assignments.as_array();
    let (n_rows, k) = a.dim();
    if n_rows == 0 || k == 0 {
        return Err(py_value_error(
            "sae_manifold_assignment_summary: assignments must be non-empty".into(),
        ));
    }
    let n_entries = n_rows.checked_mul(k).ok_or_else(|| {
        py_value_error("sae_manifold_assignment_summary: assignments shape is too large".into())
    })?;
    let mut active_total = 0_usize;
    let mut mass_total = 0.0_f64;
    for row in 0..n_rows {
        for col in 0..k {
            let assignment = a[[row, col]];
            if !assignment.is_finite() {
                return Err(py_value_error(format!(
                    "sae_manifold_assignment_summary: non-finite assignment at ({row}, {col})"
                )));
            }
            mass_total += assignment;
            if !mass_total.is_finite() {
                return Err(py_value_error(
                    "sae_manifold_assignment_summary: assignment mass overflowed".into(),
                ));
            }
            if assignment >= threshold {
                active_total += 1;
            }
        }
    }
    let avg_active = active_total as f64 / n_rows as f64;
    let mean_mass = mass_total / n_entries as f64;
    Ok((avg_active, mean_mass))
}

#[pyfunction(signature = (x, w_gate, w_amp))]
fn gated_sae_decode<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w_gate: PyReadonlyArray2<'py, f64>,
    w_amp: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let decoder = GatedSAEDecoder::new(w_gate.as_array().to_owned(), w_amp.as_array().to_owned())
        .map_err(py_value_error)?;
    let out = decoder.decode_batch(x.as_array()).map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Forward of the per-feature scalar-gate decoder (no swap).
///
/// Returns `X̂[i, d] = Σ_f gate[f] · z[i, f] · weights[d, f] + bias[d]`.
/// `bias` may be `None`. The forward and its analytic gradients are shared
/// across the Rust library, the CLI, and the PyTorch bridge via the
/// `gam::terms::decoders::interchange_decoder` primitive.
#[pyfunction(signature = (z, weights, gate, bias = None))]
fn interchange_decode_forward<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    bias: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let bias_view = bias.as_ref().map(|b| b.as_array());
    let out = core_interchange_decode_forward(CoreInterchangeDecodeForward {
        z: z.as_array(),
        weights: weights.as_array(),
        gate: gate.as_array(),
        bias: bias_view,
    })
    .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward of the per-feature scalar-gate decoder.
///
/// Returns `(grad_z, grad_weights, grad_gate, grad_bias_or_none)`.
#[pyfunction(signature = (z, weights, gate, grad_out, with_bias))]
fn interchange_decode_backward<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    grad_out: PyReadonlyArray2<'py, f64>,
    with_bias: bool,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Option<Py<PyArray1<f64>>>,
)> {
    let adjoint = core_interchange_decode_backward(
        z.as_array(),
        weights.as_array(),
        gate.as_array(),
        grad_out.as_array(),
        with_bias,
    )
    .map_err(py_value_error)?;
    let grad_bias = adjoint.grad_bias.map(|b| b.into_pyarray(py).unbind());
    Ok((
        adjoint.grad_z.into_pyarray(py).unbind(),
        adjoint.grad_weights.into_pyarray(py).unbind(),
        adjoint.grad_gate.into_pyarray(py).unbind(),
        grad_bias,
    ))
}

/// Forward of the masked-swap interchange decoder.
///
/// `mask` is a 1-D bool array of length F. For atoms with `mask[f] == true`
/// the corresponding column of `z_a` is used; otherwise the column of `z_b`.
/// Reconstruction weights and gate are shared.
#[pyfunction(signature = (z_a, z_b, mask, weights, gate, bias = None))]
fn interchange_swap_forward<'py>(
    py: Python<'py>,
    z_a: PyReadonlyArray2<'py, f64>,
    z_b: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    bias: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let bias_view = bias.as_ref().map(|b| b.as_array());
    let out = core_interchange_swap_forward(CoreInterchangeSwapForward {
        z_a: z_a.as_array(),
        z_b: z_b.as_array(),
        mask: mask.as_array(),
        weights: weights.as_array(),
        gate: gate.as_array(),
        bias: bias_view,
    })
    .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward of the masked-swap interchange decoder.
///
/// Returns `(grad_z_a, grad_z_b, grad_weights, grad_gate, grad_bias_or_none)`.
#[pyfunction(signature = (z_a, z_b, mask, weights, gate, grad_out, with_bias))]
fn interchange_swap_backward<'py>(
    py: Python<'py>,
    z_a: PyReadonlyArray2<'py, f64>,
    z_b: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
    weights: PyReadonlyArray2<'py, f64>,
    gate: PyReadonlyArray1<'py, f64>,
    grad_out: PyReadonlyArray2<'py, f64>,
    with_bias: bool,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Option<Py<PyArray1<f64>>>,
)> {
    let adjoint = core_interchange_swap_backward(
        z_a.as_array(),
        z_b.as_array(),
        mask.as_array(),
        weights.as_array(),
        gate.as_array(),
        grad_out.as_array(),
        with_bias,
    )
    .map_err(py_value_error)?;
    let grad_bias = adjoint.grad_bias.map(|b| b.into_pyarray(py).unbind());
    Ok((
        adjoint.grad_z_a.into_pyarray(py).unbind(),
        adjoint.grad_z_b.into_pyarray(py).unbind(),
        adjoint.grad_weights.into_pyarray(py).unbind(),
        adjoint.grad_gate.into_pyarray(py).unbind(),
        grad_bias,
    ))
}

/// Backward pass: compute `grad_t` and the standard REML adjoint
/// gradients at the current latent `t`.
///
/// The construction mirrors `gaussian_reml_fit_positions_backward`:
/// the inner adjoint produces `grad_x` (= ∂L/∂Φ); we then contract
/// against the N-D radial derivative jet to obtain
/// `grad_t ∈ ℝ^{n_obs × latent_dim}`.
///
/// For `grad_reml_score`, the latent contraction uses the explicit outer
/// REML formula from `/tmp/codex_outer_analytic.md` so the REML Occam
/// correction `J_i^T K_H x_i` is included with one shared solve per row.
///
/// Identifiability-mode contributions to `grad_t`:
///   * `AuxPrior`: the projected pullback of `μ · (t − ĥ(u))`;
///   * `DimSelection`: `+ Λ · t` with diagonal precision per axis.
///
/// These additive terms are computed here from the supplied auxiliary /
/// precision arrays and folded into the returned `grad_t`. The outer
/// REML loop sees a *unique* minimum because the inner Hessian on t is
/// now bounded below by `μI` (auxiliary). Fixes the audit-revised claim:
/// dim-selection/ARD alone is not a rotation-gauge fix and must be paired
/// with AuxPrior or Isometry for identifiability.
#[pyfunction(signature = (
    t,
    y,
    n_obs,
    latent_dim,
    centers,
    penalty,
    grad_lambda = 0.0,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = 0.0,
    grad_edf = 0.0,
    m = 2,
    weights = None,
    fisher_w = None,
    init_lambda = None,
    aux_u = None,
    aux_family = "ridge".to_string(),
    aux_strength = None,
    dim_selection_log_precision = None,
    basis_kind = "duchon".to_string(),
    sigma_eff_mode = "profiled".to_string(),
    tensor_knots_concat = None,
    tensor_knot_offsets = None,
    tensor_degrees = None,
))]
fn gaussian_reml_fit_latent_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: f64,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: f64,
    grad_edf: f64,
    m: usize,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    fisher_w: Option<PyReadonlyArray3<'py, f64>>,
    init_lambda: Option<f64>,
    aux_u: Option<PyReadonlyArray2<'py, f64>>,
    aux_family: String,
    aux_strength: Option<f64>,
    dim_selection_log_precision: Option<PyReadonlyArray1<'py, f64>>,
    basis_kind: String,
    sigma_eff_mode: String,
    tensor_knots_concat: Option<PyReadonlyArray1<'py, f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
    let family = match aux_family.to_ascii_lowercase().as_str() {
        "ridge" => AuxPriorFamily::Ridge,
        "linear" => AuxPriorFamily::Linear,
        other => {
            return Err(py_value_error(format!(
                "aux_family must be 'ridge' or 'linear'; got {other:?}"
            )));
        }
    };
    let sigma_eff_mode = SigmaEffMode::parse(&sigma_eff_mode).map_err(py_value_error)?;
    let basis_kind_normalized = latent_basis_kind(&basis_kind).map_err(py_value_error)?;
    let centers_view = centers.as_array();
    let t_view = t.as_array();
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();
    let effective_weights = latent_scalar_weights_with_fisher(
        n_obs,
        weights.as_ref().map(|w| w.as_array()),
        fisher_w.as_ref().map(|w| w.as_array()),
    )
    .map_err(py_value_error)?;
    let weights_view = effective_weights.as_ref().map(|w| w.view());

    // Forward design (Φ), t-matrix, and input-location jet share one dispatcher.
    let (design, t_mat, jet) = build_latent_forward_design(
        basis_kind_normalized,
        t_view,
        n_obs,
        latent_dim,
        centers_view,
        m,
        tensor_knots_concat.as_ref().map(|a| a.as_array()),
        tensor_knot_offsets.as_deref(),
        tensor_degrees.as_deref(),
        // Standalone Python backward/gradient entrypoint: no manifold/chart
        // concept here (the Rust outer optimizer routes through
        // `LatentOuterProblem`), so the latent design stays the open Euclidean
        // basis — byte-identical to prior behavior.
        None,
    )
    .map_err(py_value_error)?;
    let fit = gaussian_reml_multi_closed_form_with_cache(
        design.view(),
        y_view,
        penalty_view,
        weights_view,
        init_lambda,
        None,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    // Inner adjoint for the returned standard gradients. This still follows
    // the generic REML VJP path for y/S/w, but grad_t below replaces the
    // score-design component with the row-shared analytic latent formula.
    let backward = gaussian_reml_multi_closed_form_backward_from_fit(
        design.view(),
        y_view,
        penalty_view,
        weights_view,
        &fit,
        grad_lambda,
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score,
        grad_edf,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    let backward_for_t = if grad_reml_score != 0.0 {
        gaussian_reml_multi_closed_form_backward_from_fit(
            design.view(),
            y_view,
            penalty_view,
            weights_view,
            &fit,
            grad_lambda,
            grad_coefficients.as_ref().map(|g| g.as_array()),
            grad_fitted.as_ref().map(|g| g.as_array()),
            0.0,
            grad_edf,
        )
        .map_err(|err| py_value_error(err.to_string()))?
    } else {
        backward.clone()
    };
    let grad_x = &backward_for_t.grad_x;
    let mut grad_t =
        contract_input_loc_gradient(grad_x.view(), &jet).map_err(basis_error_to_pyerr)?;
    if grad_reml_score != 0.0 {
        add_latent_outer_reml_score_gradient(
            &mut grad_t,
            grad_reml_score,
            design.view(),
            y_view,
            t_mat.view(),
            &jet,
            penalty_view,
            weights_view,
            &fit,
            sigma_eff_mode,
        )
        .map_err(py_value_error)?;
    }
    // Identifiability-mode additive contributions to grad_t plus log-normalizer
    // adjoints. Fixes audit-revised claim that REML ARD/AuxPrior selection
    // needs the normalized prior terms, not only raw quadratic gradients.
    let mut grad_aux_log_strength: Option<f64> = None;
    let mut grad_dim_selection_log_precision: Option<Array1<f64>> = None;
    if let Some(u_arr) = aux_u.as_ref() {
        let u_view = u_arr.as_array();
        let stats = latent_aux_prior_stats(t_mat.view(), u_view, family, aux_strength)
            .map_err(py_value_error)?;
        let residual = &t_mat - &stats.targets;
        let projected_residual =
            aux_prior_targets(residual.view(), u_view, family).map_err(py_value_error)?;
        let grad_base = residual - projected_residual;
        for n in 0..n_obs {
            for a in 0..latent_dim {
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] +=
                        grad_reml_score * stats.strength.mu * grad_base[[n, a]];
                }
            }
        }
        grad_aux_log_strength = Some(
            grad_reml_score
                * (0.5 * stats.strength.mu * stats.residual_sq - 0.5 * (n_obs * latent_dim) as f64),
        );
    }
    if let Some(log_prec) = dim_selection_log_precision.as_ref() {
        let lp = log_prec.as_array();
        if lp.len() != latent_dim {
            return Err(py_value_error(format!(
                "dim_selection_log_precision length {} must equal latent_dim {}",
                lp.len(),
                latent_dim
            )));
        }
        let mut grad_log_prec = Array1::<f64>::zeros(latent_dim);
        for n in 0..n_obs {
            for a in 0..latent_dim {
                let prec = lp[a].exp();
                if grad_reml_score != 0.0 {
                    grad_t[n * latent_dim + a] += grad_reml_score * prec * t_mat[[n, a]];
                }
            }
        }
        for a in 0..latent_dim {
            let log_alpha = lp[a];
            let prec = log_alpha.exp();
            if !(prec.is_finite() && prec > 0.0) {
                return Err(py_value_error(format!(
                    "dim_selection_log_precision[{a}] must exponentiate to a finite positive precision"
                )));
            }
            let mut sq = 0.0_f64;
            for n in 0..n_obs {
                let v = t_mat[[n, a]];
                sq += v * v;
            }
            grad_log_prec[a] = grad_reml_score * (0.5 * prec * sq - 0.5 * (n_obs as f64));
        }
        grad_dim_selection_log_precision = Some(grad_log_prec);
    }
    let mut grad_t_matrix = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            grad_t_matrix[[n, a]] = grad_t[n * latent_dim + a];
        }
    }
    let out = PyDict::new(py);
    out.set_item("grad_t", grad_t_matrix.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_penalty", backward.grad_penalty.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad) = grad_aux_log_strength {
        out.set_item("grad_aux_log_strength", grad)?;
        out.set_item("grad_log_mu", grad)?;
    } else {
        out.set_item("grad_aux_log_strength", py.None())?;
        out.set_item("grad_log_mu", py.None())?;
    }
    if let Some(grad) = grad_dim_selection_log_precision {
        out.set_item("grad_dim_selection_log_precision", grad.into_pyarray(py))?;
    } else {
        out.set_item("grad_dim_selection_log_precision", py.None())?;
    }
    Ok(out.unbind())
}

/// Owned inputs for the latent outer-optimization objective.
///
/// Bundles the data the value/gradient evaluation needs so a single struct can
/// be reused across trust-region iterations and restarts without re-copying
/// from Python.
struct LatentOuterProblem {
    y: Array2<f64>,
    centers: Array2<f64>,
    penalty: Array2<f64>,
    weights: Option<Array1<f64>>,
    aux_u: Option<Array2<f64>>,
    dim_selection: Option<Array1<f64>>,
    family: AuxPriorFamily,
    aux_strength: Option<f64>,
    init_lambda: Option<f64>,
    sigma_eff_mode: SigmaEffMode,
    n_obs: usize,
    latent_dim: usize,
    m: usize,
    basis_kind: String,
    tensor_knots: Option<Array1<f64>>,
    tensor_knot_offsets: Option<Vec<usize>>,
    tensor_degrees: Option<Vec<usize>>,
    /// Per-axis chart period of the optimizer's manifold (radians) for the
    /// Duchon decoder; `None` on Euclidean / sphere so the decoder stays the
    /// open Euclidean basis. Derived from the `manifold` string in
    /// `gaussian_reml_optimize_latent` via `latent_manifold_periodic_descriptor`.
    periodic: Option<Vec<Option<f64>>>,
}

impl LatentOuterProblem {
    /// REML score (inner Gaussian REML plus aux/dim identifiability priors) and,
    /// when `want_grad`, the outer latent gradient `∂(reml_score)/∂t`.
    ///
    /// The value reproduces [`gaussian_reml_fit_latent`]'s `reml_score` and the
    /// gradient reproduces [`gaussian_reml_fit_latent_backward`]'s `grad_t` at
    /// `grad_reml_score = 1`, so the optimizer descends exactly the quantity the
    /// forward primitive reports. A non-finite or unsolvable configuration maps
    /// to `+∞` with no gradient, which the trust region rejects rather than
    /// propagating a NaN into the inner adjoint.
    fn value_and_grad(
        &self,
        t_flat: ArrayView1<'_, f64>,
        want_grad: bool,
    ) -> (f64, Option<Array1<f64>>) {
        match self.try_value_and_grad(t_flat, want_grad) {
            Ok(pair) => pair,
            Err(_) => (f64::INFINITY, None),
        }
    }

    fn try_value_and_grad(
        &self,
        t_flat: ArrayView1<'_, f64>,
        want_grad: bool,
    ) -> Result<(f64, Option<Array1<f64>>), String> {
        let (design, t_mat, jet) = build_latent_forward_design(
            &self.basis_kind,
            t_flat,
            self.n_obs,
            self.latent_dim,
            self.centers.view(),
            self.m,
            self.tensor_knots.as_ref().map(|a| a.view()),
            self.tensor_knot_offsets.as_deref(),
            self.tensor_degrees.as_deref(),
            self.periodic.as_deref(),
        )?;
        let weights_view = self.weights.as_ref().map(|w| w.view());
        let fit = gaussian_reml_multi_closed_form_with_cache(
            design.view(),
            self.y.view(),
            self.penalty.view(),
            weights_view,
            self.init_lambda,
            None,
        )
        .map_err(|err| err.to_string())?;
        let (prior_score, _aux_state) = latent_prior_score_and_aux_state_for_t(
            t_mat.view(),
            self.aux_u.as_ref().map(|a| a.view()),
            self.family,
            self.aux_strength,
            self.dim_selection.as_ref().map(|a| a.view()),
        )?;
        let value = fit.reml_score + prior_score;
        if !value.is_finite() {
            return Ok((f64::INFINITY, None));
        }
        if !want_grad {
            return Ok((value, None));
        }
        let mut grad_t = Array1::<f64>::zeros(self.n_obs * self.latent_dim);
        add_latent_outer_reml_score_gradient(
            &mut grad_t,
            1.0,
            design.view(),
            self.y.view(),
            t_mat.view(),
            &jet,
            self.penalty.view(),
            weights_view,
            &fit,
            self.sigma_eff_mode,
        )?;
        // Identifiability-prior contributions, identical to the backward path's
        // grad_t assembly at `grad_reml_score = 1`.
        if let Some(u_arr) = self.aux_u.as_ref() {
            let u_view = u_arr.view();
            let stats =
                latent_aux_prior_stats(t_mat.view(), u_view, self.family, self.aux_strength)?;
            let residual = &t_mat - &stats.targets;
            let projected_residual = aux_prior_targets(residual.view(), u_view, self.family)?;
            let grad_base = residual - projected_residual;
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    grad_t[n * self.latent_dim + a] += stats.strength.mu * grad_base[[n, a]];
                }
            }
        }
        if let Some(log_prec) = self.dim_selection.as_ref() {
            for n in 0..self.n_obs {
                for a in 0..self.latent_dim {
                    let prec = log_prec[a].exp();
                    grad_t[n * self.latent_dim + a] += prec * t_mat[[n, a]];
                }
            }
        }
        if !grad_t.iter().all(|value| value.is_finite()) {
            return Ok((f64::INFINITY, None));
        }
        Ok((value, Some(grad_t)))
    }
}

/// Adapter exposing [`LatentOuterProblem`] to the Riemannian trust region.
struct LatentOuterObjective<'a> {
    problem: &'a LatentOuterProblem,
}

impl gam::geometry::RiemannianObjective for LatentOuterObjective<'_> {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::geometry::GeometryResult<(f64, Array1<f64>)> {
        // A degenerate point yields `+∞` and a zero gradient: the trust region
        // reads a zero gradient at the start as "stationary" (it stops at the
        // finite init) and a `+∞` trial value as a rejected step (it shrinks).
        match self.problem.value_and_grad(point, true) {
            (value, Some(grad)) => Ok((value, grad)),
            (_, None) => Ok((f64::INFINITY, Array1::<f64>::zeros(point.len()))),
        }
    }
}

/// Build the manifold the outer optimizer walks `t` on. `manifold` names the
/// per-observation geometry; the full latent lives on the `n_obs`-fold product.
/// Per-axis chart period for the latent decoder, derived from the optimizer's
/// manifold so the Duchon decoder is a genuine function ON that manifold.
///
/// The circle manifold (`src/geometry/circle.rs`) wraps each coordinate to
/// `[-π, π)`, i.e. period `2π = TAU` radians; the torus is its `d`-fold product.
/// The optimizer retracts the latent in radians on these charts, and the
/// periodic eigenmap seed (`latent_periodic_seed_start`) also produces radians,
/// so the decoder kernel distance must be measured modulo `TAU` per circular
/// axis and satisfy `Φ(θ) = Φ(θ + TAU)`. A non-periodic axis is `None`.
///
/// Euclidean / sphere return `None` (no axis is a circle): those latent fits
/// stay byte-identical to the open Euclidean Duchon basis. (`sphere` is `S^{d-1}`
/// embedded in `R^d` with NO periodic chart axis here — the spherical structure
/// is carried by the retraction, not by a per-axis wrap.)
fn latent_manifold_periodic_descriptor(
    manifold: &str,
    latent_dim: usize,
) -> Option<Vec<Option<f64>>> {
    match manifold.to_ascii_lowercase().replace('-', "_").as_str() {
        "circle" | "s1" if latent_dim == 1 => Some(vec![Some(std::f64::consts::TAU)]),
        "torus" => Some(vec![Some(std::f64::consts::TAU); latent_dim]),
        _ => None,
    }
}

fn build_latent_outer_manifold(
    manifold: &str,
    n_obs: usize,
    latent_dim: usize,
) -> Result<Box<dyn gam::geometry::RiemannianManifold>, String> {
    let per_point = match manifold.to_ascii_lowercase().replace('-', "_").as_str() {
        "euclidean" | "rn" => {
            // One flat Euclidean block over the whole latent is equivalent to
            // the product and avoids the per-observation slicing overhead.
            return Ok(Box::new(gam::geometry::EuclideanManifold::new(
                n_obs * latent_dim,
            )));
        }
        "circle" | "s1" => {
            if latent_dim != 1 {
                return Err(format!(
                    "circle latent manifold requires latent_dim == 1; got {latent_dim}"
                ));
            }
            gam::geometry::ManifoldSpec::Circle
        }
        "sphere" => {
            if latent_dim < 2 {
                return Err(format!(
                    "sphere latent manifold requires latent_dim >= 2 (S^{{d-1}} embeds in R^d); got {latent_dim}"
                ));
            }
            gam::geometry::ManifoldSpec::Sphere {
                intrinsic_dim: latent_dim - 1,
            }
        }
        "torus" => gam::geometry::ManifoldSpec::Torus { dim: latent_dim },
        other => {
            return Err(format!(
                "unknown latent manifold {other:?}; expected one of euclidean|circle|sphere|torus"
            ));
        }
    };
    let parts = std::iter::repeat_with(|| per_point.clone())
        .take(n_obs)
        .collect();
    gam::geometry::ManifoldSpec::Product(parts)
        .build()
        .map_err(|err| err.to_string())
}

/// Build the restart-0 start for the latent outer optimizer from a spectral
/// (Laplacian-eigenmaps) embedding of the responses `y`.
///
/// The embedding recovers the intrinsic coordinate up to monotone/rotation
/// gauge; each axis is then affinely mapped from `[0, 1]` onto the span of the
/// decoder `centers` for that axis so the seed lands where the basis `Φ` is
/// well-conditioned. On a *periodic* latent manifold (circle/torus) the natural
/// seed is the circular coordinate recovered from the leading Laplacian modes
/// (see [`latent_periodic_seed_start`]); the sphere has no closed-form spectral
/// seed here, so the caller's `t` is used unchanged.
///
/// A spread seed is essential, not optional: the outer optimizer's REML
/// objective is degenerate at a *collapsed* latent (all rows at the same
/// coordinate give identical decoder rows → a rank-deficient inner solve and no
/// usable descent direction). The default caller start is the all-zero vector,
/// which on a periodic manifold is exactly the collapsed configuration; without
/// a spread seed the circle/torus optimizer can never escape it (issue #876).
fn latent_spectral_seed_start(
    y: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    manifold: &str,
    n_obs: usize,
    latent_dim: usize,
    seed_neighbors: usize,
    caller_t: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let manifold_norm = manifold.to_ascii_lowercase().replace('-', "_");
    if matches!(manifold_norm.as_str(), "circle" | "s1" | "torus") {
        return latent_periodic_seed_start(y, n_obs, latent_dim, seed_neighbors, caller_t);
    }
    if !matches!(manifold_norm.as_str(), "euclidean" | "rn") {
        return Ok(caller_t.to_owned());
    }
    if y.nrows() != n_obs {
        return Err(format!(
            "spectral seed: y has {} rows but n_obs = {n_obs}",
            y.nrows()
        ));
    }
    // Too few rows to expose `latent_dim` non-trivial modes: fall back to the
    // caller's start rather than failing the whole optimize call.
    if n_obs < latent_dim + 2 {
        return Ok(caller_t.to_owned());
    }
    let coords = gam::geometry::laplacian_eigenmap_coords(y, latent_dim, seed_neighbors)?;
    // Per-axis target span from the decoder centers; fall back to [0, 1] when an
    // axis has no corresponding center column or a degenerate span.
    let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
    for a in 0..latent_dim {
        let (lo, hi) = if a < centers.ncols() {
            let col = centers.column(a);
            let lo = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lo.is_finite() && hi.is_finite() && hi > lo {
                (lo, hi)
            } else {
                (0.0, 1.0)
            }
        } else {
            (0.0, 1.0)
        };
        for n in 0..n_obs {
            start[n * latent_dim + a] = lo + coords[[n, a]] * (hi - lo);
        }
    }
    Ok(start)
}

/// Spectral seed for a *periodic* latent (circle / torus), returning each row's
/// angle in `[-π, π)` per axis.
///
/// On a circle the two leading non-trivial Laplacian-eigenmap modes of the
/// responses are (up to rotation/reflection — exactly the circle's gauge) the
/// `cos θ` / `sin θ` pair of the intrinsic angle, so `θ = atan2(sin-mode,
/// cos-mode)` recovers the circular coordinate directly. A torus of dimension
/// `d` is `d` independent circles; we recover one angle per axis from its own
/// pair of modes, requesting `2·d` modes from the embedding and pairing them in
/// order. The recovered angle is a *seed* — correct up to the periodic gauge the
/// decoder is free in — that the Riemannian outer optimizer then polishes.
///
/// Crucially this seed is *spread* around the circle, breaking the collapsed
/// all-zero start (issue #876). When there are too few rows to expose `2·d`
/// non-trivial modes the embedding cannot run; rather than start collapsed we
/// fall back to a deterministic equispaced angular sweep on each axis, which is
/// still non-degenerate (distinct decoder rows) so the optimizer has a usable
/// gradient.
fn latent_periodic_seed_start(
    y: ArrayView2<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    seed_neighbors: usize,
    caller_t: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    use std::f64::consts::TAU;

    if latent_dim == 0 {
        return Ok(caller_t.to_owned());
    }
    if y.nrows() != n_obs {
        return Err(format!(
            "periodic spectral seed: y has {} rows but n_obs = {n_obs}",
            y.nrows()
        ));
    }
    // A circle/torus axis needs two embedding modes (cos/sin); recover one angle
    // per axis. If the caller already supplied a *spread* warm start (not the
    // collapsed default), keep it — the optimizer can polish a good start, but it
    // can never escape a collapsed one. "Spread" is measured per axis by the
    // angular range the wrapped coordinates cover.
    let caller_spread = caller_t.len() == n_obs * latent_dim
        && (0..latent_dim).any(|a| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for n in 0..n_obs {
                let v = wrap_to_pi(caller_t[n * latent_dim + a]);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            (hi - lo) > 1.0e-6
        });
    if caller_spread {
        return Ok(caller_t.to_owned());
    }

    let modes = 2 * latent_dim;
    // `laplacian_eigenmap_coords` needs `n >= modes + 2` rows to expose `modes`
    // non-trivial eigenvectors. With fewer rows, sweep angles deterministically.
    if n_obs < modes + 2 {
        let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
        for a in 0..latent_dim {
            for n in 0..n_obs {
                let frac = if n_obs > 0 {
                    n as f64 / n_obs as f64
                } else {
                    0.0
                };
                start[n * latent_dim + a] = wrap_to_pi(frac * TAU);
            }
        }
        return Ok(start);
    }

    // The raw (un-rescaled) generalized eigenvectors are what carry the cos/sin
    // structure; `laplacian_eigenmap_coords` already rescales each axis to
    // [0, 1], which destroys the relative sign/scale needed for atan2. We
    // instead read `2·d` modes and undo the per-axis affine map by recentering
    // each mode to zero mean before pairing — the rescale is affine per mode, so
    // recentering recovers the angle up to the same rotation gauge.
    let coords = gam::geometry::laplacian_eigenmap_coords(y, modes, seed_neighbors)?;
    let mut mode_mean = vec![0.0f64; modes];
    for a in 0..modes {
        let mut sum = 0.0;
        for n in 0..n_obs {
            sum += coords[[n, a]];
        }
        mode_mean[a] = sum / n_obs as f64;
    }

    let mut start = Array1::<f64>::zeros(n_obs * latent_dim);
    for axis in 0..latent_dim {
        let cos_mode = 2 * axis;
        let sin_mode = 2 * axis + 1;
        for n in 0..n_obs {
            let c = coords[[n, cos_mode]] - mode_mean[cos_mode];
            let s = coords[[n, sin_mode]] - mode_mean[sin_mode];
            let angle = if c == 0.0 && s == 0.0 {
                // Degenerate row (both modes vanish): place it deterministically
                // around the circle so it does not coincide with its neighbours.
                wrap_to_pi((n as f64 / n_obs as f64) * TAU)
            } else {
                s.atan2(c)
            };
            start[n * latent_dim + axis] = angle;
        }
        // Guard against a collapsed axis (both modes constant → all angles
        // equal): fall back to an equispaced sweep on that axis only.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for n in 0..n_obs {
            let v = start[n * latent_dim + axis];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if !(hi - lo > 1.0e-6) {
            for n in 0..n_obs {
                let frac = if n_obs > 0 {
                    n as f64 / n_obs as f64
                } else {
                    0.0
                };
                start[n * latent_dim + axis] = wrap_to_pi(frac * TAU);
            }
        }
    }
    Ok(start)
}

/// Wrap an angle to the half-open interval `[-π, π)`.
fn wrap_to_pi(angle: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let mut a = angle % TAU;
    if a >= PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

/// Optimize the latent coordinate `t` against the Gaussian-REML objective.
///
/// Unlike [`gaussian_reml_fit_latent`], which performs a single `β | t` inner
/// solve at a fixed `t`, this routine runs the *outer* latent optimization: it
/// minimizes the REML score over `t` with a Riemannian trust region driven by
/// the analytic `∂(reml_score)/∂t` (the same gradient
/// [`gaussian_reml_fit_latent_backward`] returns), retracting each accepted
/// step onto `manifold`. It returns the full REML fit dictionary *at the
/// converged latent* plus the optimized `t`/`latent` arrays.
///
/// The latent REML objective is non-convex (a GP-LVM-style coordinate problem),
/// so a single cold random start may settle in a poor local optimum. By default
/// (`init="spectral"`) restart 0 starts from a Laplacian-eigenmaps embedding of
/// the responses, which recovers the intrinsic coordinate up to gauge and lets
/// the optimizer polish it to the global fit instead of sorting rows from
/// scratch; the passed-in `t` is then only a fallback (too few rows, or a
/// non-Euclidean `manifold`). Pass `init="caller"` to start from `t` unchanged
/// (a pure local solve / explicit warm start), and `n_restarts > 1` to also
/// optimize from perturbed starts and keep the lowest-score result.
///
/// Shift-invariant relative-gradient stationarity measure for the latent outer
/// solve: `‖∇ₜ f(t̂)‖_g / max(‖∇ₜ f(t₀)‖_g, 1)`, comparing the projected
/// Riemannian gradient norm at the chosen latent to the gradient norm at the
/// INITIAL iterate `t₀`. This is the FFI analogue of `relative_stationarity` in
/// `src/geometry/optimizer.rs` (issue #954), kept byte-for-byte identical to it
/// so the diagnostic `converged` flag agrees with the optimizer's own stopping
/// rule:
///
/// * **Shift-invariant** — the objective value `f` does not enter at all, so an
///   additive shift `f → f + C` (which leaves the minimizer, gradient, Hessian,
///   and model reduction unchanged) cannot move the measure. The earlier
///   `‖∇ₜ f‖·‖t‖_typ / max(|f|, 1)` divided by the objective magnitude, so a
///   large `C` inflated the denominator and could falsely certify a
///   non-stationary latent as converged (#954).
/// * **Scale-invariant** — under `f → c·f` both `‖∇ₜ f(t̂)‖` and `‖∇ₜ f(t₀)‖`
///   scale by `c`, so the ratio is unchanged and a fixed `grad_tol` reads as a
///   true *relative* tolerance.
/// * **#879 O(n) calibration** — the profiled REML objective leaves `‖∇ₜ f‖` at
///   an O(n) magnitude even at a genuine stationary point near interpolation;
///   anchoring to `‖∇ₜ f(t₀)‖` (itself O(n)) divides that magnitude out, while
///   the `max(·, 1)` floor reduces the test to the bare absolute
///   `‖∇ₜ f‖ ≤ grad_tol` on a unit-scale objective.
/// * **Non-finite** — a blown-up iterate (`‖∇ₜ f‖` or `‖∇ₜ f(t₀)‖` not finite)
///   maps to `+∞`, so it is never reported stationary.
fn latent_relative_stationarity(grad_norm: f64, grad0_norm: f64) -> f64 {
    if !grad_norm.is_finite() || !grad0_norm.is_finite() {
        return f64::INFINITY;
    }
    grad_norm / grad0_norm.max(1.0)
}

#[cfg(test)]
mod sae_euclidean_oos_rebuild_tests {
    use super::{monomial_exponents, sae_euclidean_degree_for_basis_size};

    /// #1132 bug 3: the OOS basis rebuild for a Euclidean (linear) atom must
    /// re-emit a basis whose width `M` equals the TRAINED decoder block's row
    /// count. The width is `monomial_exponents(dim, degree).len()` where `dim`
    /// is the build dimension (`centers.ncols()`). Recovering the degree from
    /// `(dim, trained_M)` and rebuilding against the same `dim` must therefore
    /// reproduce `trained_M` exactly. The regression case is a 1-D linear atom
    /// whose trained decoder has `M = 1` (degree 0): the recovery must yield
    /// degree 0 and width 1, NOT re-expand to width 3 (degree 2) — the
    /// "decoder_blocks[0] has M=1 but rebuilt basis has M=3" OOS failure.
    fn rebuilt_m_for(dim: usize, trained_m: usize) -> usize {
        let degree = sae_euclidean_degree_for_basis_size(dim, trained_m)
            .expect("degree must be recoverable from the trained decoder width");
        monomial_exponents(dim, degree).len()
    }

    #[test]
    fn euclidean_oos_rebuild_m_matches_trained_decoder_m() {
        // The exact #1132 regression: dim = 1, trained M = 1 (constant-only).
        assert_eq!(
            rebuilt_m_for(1, 1),
            1,
            "1-D linear atom with decoder M=1 must rebuild to M=1, not M=3"
        );
        // The recovered width must equal the trained M across the supported
        // degrees and dimensions (self-consistency of the decoder-anchored
        // recovery the OOS / steer paths now use).
        for dim in 1..=2usize {
            for degree in 0..=2usize {
                let trained_m = monomial_exponents(dim, degree).len();
                assert_eq!(
                    rebuilt_m_for(dim, trained_m),
                    trained_m,
                    "dim={dim}, degree={degree}: rebuilt M must equal trained M"
                );
            }
        }
    }
}

#[cfg(test)]
mod sae_assignment_kind_tests {
    use super::canonicalize_assignment_kind;

    /// #1777 — the FFI assignment-kind parser EMITS the primary "threshold_gate"
    /// token and ACCEPTS both it and the legacy "jumprelu" alias, mapping both to
    /// the renamed `AssignmentMode::ThresholdGate`. "softmax" / "ibp_map" pass
    /// through unchanged; any other token is a caller error.
    #[test]
    fn threshold_gate_accepts_both_spellings_and_emits_primary() {
        // Both the primary spelling and the legacy alias canonicalize to the same
        // emitted token.
        assert_eq!(
            canonicalize_assignment_kind("threshold_gate").unwrap(),
            "threshold_gate"
        );
        assert_eq!(
            canonicalize_assignment_kind("jumprelu").unwrap(),
            "threshold_gate",
            "the legacy 'jumprelu' alias must map to the renamed variant's token"
        );
        // The other families pass through unchanged.
        assert_eq!(canonicalize_assignment_kind("softmax").unwrap(), "softmax");
        assert_eq!(canonicalize_assignment_kind("ibp_map").unwrap(), "ibp_map");
        // An unknown token errors, and the message names the primary spelling
        // while still advertising the accepted legacy alias.
        let err = canonicalize_assignment_kind("bogus").unwrap_err();
        assert!(
            err.contains("threshold_gate") && err.contains("jumprelu"),
            "error must name the primary token and the legacy alias; got {err:?}"
        );
    }
}

#[cfg(test)]
mod sae_linear_atom_tests {
    use super::{sae_atom_basis_kind_from_str, sae_atom_basis_kind_name};
    use gam::terms::sae::manifold::{EuclideanPatchEvaluator, SaeAtomBasisKind, SaeBasisEvaluator};
    use ndarray::Array2;

    /// #1221 — `"linear"` (and its synonyms) is a first-class topology distinct
    /// from `"euclidean"`/`"euclidean_patch"` (the degree-2 quadratic patch), and
    /// it round-trips under the honest name `"linear"`. The quadratic patch keeps
    /// its own `"euclidean_patch"` name and additionally accepts the explicit
    /// `"euclidean_quadratic_patch"` synonym.
    #[test]
    fn linear_topology_is_first_class_and_round_trips() {
        for name in ["linear", "linear_rank1", "affine", "LINEAR"] {
            assert_eq!(
                sae_atom_basis_kind_from_str(name),
                SaeAtomBasisKind::Linear,
                "{name:?} must parse to the genuinely-linear atom"
            );
        }
        assert_eq!(
            sae_atom_basis_kind_name(&SaeAtomBasisKind::Linear),
            "linear",
            "the linear atom must round-trip under its honest name"
        );
        // The quadratic patch is a DIFFERENT kind — `"linear"` must not collapse
        // onto it, or the curved-vs-linear comparison would be mislabeled again.
        for name in ["euclidean", "euclidean_patch", "euclidean_quadratic_patch"] {
            assert_eq!(
                sae_atom_basis_kind_from_str(name),
                SaeAtomBasisKind::EuclideanPatch,
                "{name:?} is the degree-2 quadratic patch, distinct from linear"
            );
        }
    }

    /// #1221 — the genuinely-linear atom's decoder reconstructs the affine image
    /// `γ(t) = b₀ + t·b₁` EXACTLY. Its evaluator is the degree-1 monomial patch
    /// `Φ(t) = [1, t]` (width `d + 1 = 2` at `d = 1`), so for a decoder
    /// `B = [[b₀…], [b₁…]]` the reconstruction `Φ(t)·B` equals `b₀ + t·b₁` to
    /// machine precision — the property the reconstruction-parity baseline needs.
    #[test]
    fn linear_atom_reconstructs_affine_image_exactly() {
        let evaluator = EuclideanPatchEvaluator::new(1, 1).expect("degree-1 patch");
        assert_eq!(
            evaluator.basis_size(),
            2,
            "a degree-1 (linear/affine) patch in 1-D has width 2: {{1, t}}"
        );
        // Decoder over p = 2 output channels: γ(t) = b0 + t·b1.
        let b0 = [0.7_f64, -1.3];
        let b1 = [2.0_f64, 0.5];
        let decoder =
            Array2::from_shape_vec((2, 2), vec![b0[0], b0[1], b1[0], b1[1]]).expect("2x2 decoder");

        let coords =
            Array2::from_shape_vec((5, 1), vec![-2.0, -0.5, 0.0, 1.0, 3.0]).expect("5x1 coords");
        let (phi, _jet) = evaluator
            .evaluate(coords.view())
            .expect("evaluate linear patch");
        assert_eq!(phi.dim(), (5, 2));
        let recon = phi.dot(&decoder); // (5, 2) = Φ·B
        for (row, &t) in coords.column(0).iter().enumerate() {
            for ch in 0..2 {
                let expected = b0[ch] + t * b1[ch];
                assert!(
                    (recon[[row, ch]] - expected).abs() < 1e-12,
                    "linear atom must reconstruct b0 + t·b1 exactly: \
                     row {row} ch {ch} got {} want {expected}",
                    recon[[row, ch]]
                );
            }
        }
    }
}

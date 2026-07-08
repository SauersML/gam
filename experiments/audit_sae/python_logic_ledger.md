# gamfit Python-logic SPEC audit ledger

**SPEC rule audited:** "Python should be a thin wrapper over Rust, and should avoid containing math or much logic. Exception to both: if Python logic is necessary for interaction with external software, e.g., PyTorch. Exception: analysis of results or examples in the proper, non-production directories."

**Scope:** every `.py` under `/Users/user/gam/gamfit/` including `gamfit/torch/`. Exempt (not audited): `bench/`, `experiments/`, `tests/`, and `gamfit/examples/`.

**Method:** all ~90 production modules read in full (the large ones — `_sae_manifold.py`, `torch/manifold_sae.py`, `_api.py`, `_select_topology.py`, `torch/_basis.py`, `torch/penalties.py`, `torch/fit.py` — read line-by-line). Precedent applied for the torch/JAX exception: autograd-tape plumbing and tensor movement qualify; math that could be a Rust kernel behind the FFI does not (per the migration of `sae_sinkhorn_balance_bias` to Rust and the 2026-07-08 deletion of a Python coordinate-projection solver from `torch/manifold_sae.py`).

---

## Summary

| Severity | Count |
|---|---|
| CORE-MATH (numerical results depend on it) | 23 |
| MODEL-LOGIC (behavioral decisions) | 20 |
| BORDERLINE (marshalling-adjacent) | 32 |
| **Total** | **75** |

### Top-10 worst offenders (clearest, highest-impact violations)

| # | file:line | what | severity |
|---|---|---|---|
| 1 | `torch/manifold_sae.py:1096-1317` | full deterministic-annealing EM routing/assignment solver (`reconstruction_topk_gate`) in torch | MODEL-LOGIC + CORE-MATH |
| 2 | `torch/penalties.py:619-812` | four SAE activation kernels + analytic Jacobians (`_JumpReLUSTEFn`, `_IBPMapFn`, `_JumpReLUBoundedGateFn`, `_TopKActivationFn`), self-admitted transcriptions of Rust | CORE-MATH |
| 3 | `torch/manifold_sae.py:947-1068` | union-of-subspaces model-selection SVD search (`_quadratic_subspace_anchor`) | MODEL-LOGIC + CORE-MATH |
| 4 | `torch/_basis.py:806-860` | closed-form ridge VJP `(XᵀWX+λS)⁻¹` assembled in torch (`_gwr_vjp`) | CORE-MATH |
| 5 | `torch/geometry.py:33-271` | full Aitchison/log-ratio compositional-geometry stack (CLR/ILR/ALR/Fréchet/log-exp) | CORE-MATH |
| 6 | `torch/manifold_sae.py:1319-1431` | residual-PC matching-pursuit commitment solver (`_matching_pursuit_commit`) | MODEL-LOGIC + CORE-MATH |
| 7 | `torch/modules.py:87-150` | bespoke hand-derived adaptive-top-k STE estimator + VJP (`_AdaptiveTopKSTE`) | CORE-MATH |
| 8 | `_select_topology.py:491-609` | held-out Gaussian log-score objective matrix for topology stacking (`stack_topologies`) | CORE-MATH |
| 9 | `_select_topology.py:226-422` | model-selection budget-cascade + score composition (`select_topology`) | MODEL-LOGIC |
| 10 | `distill.py:80-110` | SAE encode-assignment kernel (softmax/IBP-MAP/jumprelu) reimplemented in numpy (`_activation_from_logits`) | CORE-MATH |

---

## CORE-MATH violations

### torch/ (PyTorch exception scrutinized and rejected — these are kernels, not tape plumbing)

- **`torch/penalties.py:619-812`** — `_JumpReLUSTEFn` (633-658), `_IBPMapFn` (674-705), `_JumpReLUBoundedGateFn` (734-760), `_TopKActivationFn` (793-812). Each computes an SAE gate/activation value **and** its analytic diagonal Jacobian in torch (sigmoid/softplus gates, truncated stick-breaking prior `π_k=(α/(α+1))^{k+1}`, slope terms). **Interop defense: FAILS, self-admittedly** — every docstring says the body is "a pure-torch, on-device transcription of the Rust source of truth" (e.g. `gam_sae::assignment::topk_activation_row_value_grad`, `jumprelu_row_value_grad`, `ibp_map_row_value_grad`) and justifies it purely on device/perf grounds ("no (N,K) matrix crosses the boundary"), which is exactly the reasoning the SPEC rejects and the same rationale that governed the `sae_sinkhorn_balance_bias` migration. **Migration:** expose the existing `crates/gam-sae/src/assignment.rs` `*_row_value_grad` kernels (and `sparsity.rs::jumprelu_gate_value_grad`) as GPU-capable pyffi entry points (or a torch custom-op/CUDA kernel); the autograd Function keeps only save/restore + `grad·jac_diag`.

- **`torch/_basis.py:806-860`** (`_gwr_vjp`; invoked by `_GaussianWeightedRidgeFn.backward` 895-904, batched loop 948-998) — the entire closed-form analytic VJP of the row-weighted ridge `β=(XᵀWX+λS)⁻¹XᵀWY`: assembles `A`, `torch.linalg.solve`, `Ā=−λ_adj βᵀ`, and four gradient terms. **Interop defense: FAILS** — the forward already round-trips to Rust (`_api.gaussian_weighted_ridge`); only the backward math lives in Python, inconsistent with every REML wrapper in `_reml.py` (which route the backward through a Rust adjoint). **Migration:** add `gaussian_weighted_ridge_backward` / `..._batch_backward` to `crates/gam-pyffi/src/`. FFI: `fn gaussian_weighted_ridge_backward(x,y,penalty,weights,coef,ridge_lambda, grad_coef, grad_fitted) -> (grad_x, grad_y, grad_penalty, grad_weights)`.

- **`torch/geometry.py:33-271`** — Aitchison/log-ratio stack: `_closure_tensor`, `clr` (76-82), `_ilr_basis` Helmert contrast construction (85-104), `ilr`/`inverse_ilr` (107-136), `aitchison_metric` (139-153), `simplex_frechet_mean` (183-194), `simplex_log_map`/`simplex_exp_map` (197-271). **Interop defense: FAILS** — a non-torch numpy implementation already exists (`gamfit._response_geometry`), and the tell-tale inconsistency is that `sphere_frechet_mean` (277-297) routes to Rust while `simplex_frechet_mean` is hand-rolled. **Migration:** back all charts with response-geometry kernels exposed through `gam-pyffi` with input-location jets (mirror `sphere_basis_jet`); torch keeps a jet-contracting autograd Function.

- **`torch/geometry.py:300-341`** — `sphere_log_map` / `sphere_exp_map` (acos, tangent projection, `θ/sinθ`, antipodal guards). **Interop defense: FAILS** — parallels `hyperbolic.py` where every Poincaré map routes to Rust `gam::geometry::poincare`, and `sphere_frechet_mean` already routes to Rust. **Migration:** `sphere_log_map`/`sphere_exp_map` FFI in the gam geometry crate with analytic jet.

- **`torch/fit.py:133-184, 368-411`** — `_marginal_bspline_design_penalty` + tensor_bspline branch: builds the row-wise Khatri-Rao tensor design and the Kronecker-sum penalty `S=Σ_a I⊗S_a⊗I` via `torch.kron`, explicitly "matching the Rust TensorBSpline penalty." **Interop defense: PARTIAL FAIL** — the 1-D marginals legitimately flow autograd to `points`, but the Khatri-Rao column tensoring and Kronecker-sum penalty assembly have no gradient to `points` and duplicate an existing Rust term. **Migration:** `tensor_bspline_design_penalty(marginals_knots, degrees, orders, points)` in the basis/penalty crate via `gam-pyffi`.

- **`torch/modules.py:87-150`** (`_AdaptiveTopKSTE`) + `_predict_k`/`forward` (265-300) — differentiable order-statistic threshold `τ_i` (sort + interpolation), sigmoid soft top-k mask, hard mask, soft count; backward hand-derives `dm/dz = σ'·sign(z)/temp`. **Interop defense: FAILS** — not even a transcription of a Rust source of truth; a bespoke numerical estimator authored in torch. The module's own `reml_descriptor` concedes "A future Rust-side analytic with rho_count==1 is required." **Migration:** adaptive-top-k value+grad kernel in `crates/gam-sae/src/assignment.rs` via pyffi.

- **`torch/manifold_sae.py:2219-2262`** — `decoder_harmonic_penalty`: `Σ_{h≥2} h⁴ (‖sin_row‖² + ‖cos_row‖²)` computed in torch. **Interop defense: FAILS** — the module contract (docstring 20-29) and sibling penalties (`decoder_ortho_penalty`, `decoder_monotonicity_penalty`) all route through Rust `analytic_penalty` descriptors; this one silently does the arithmetic in Python. **Migration:** add a `harmonic_smoothness` analytic-penalty descriptor in `crates/gam-sae/` and expose via `analytic_penalty_value_grad`; wrap like `BlockOrthogonalityPenalty`.

- **`torch/manifold_sae.py:883-945`** — `_direction_cluster_anchor`: seed-free k-lines clustering of row directions — SVD-based farthest-line init, 25 reassignment/refit iterations (`torch.linalg.svd` per cluster per iter), balance/margin gating. **Interop defense: FAILS** (`@torch.no_grad()`, no model tape). **Migration:** `sae_direction_cluster_anchor(x, n_atoms, iters) -> (onehot, confident)` in `gam-sae`.

### numpy / non-torch (exception does not apply at all)

- **`distill.py:80-110`** (`_activation_from_logits`) — SAE assignment activations in pure numpy: softmax, IBP-MAP geometric prior × sigmoid, threshold_gate/jumprelu hard-sigmoid with ±709 clip. Duplicates gam-sae encode-assignment math. **Migration:** `sae_activation_from_logits(logits, kind, tau, alpha, jumprelu_threshold)` in `crates/gam-pyffi/src/` dispatching to the gam-sae assignment enum.

- **`intervention_calibration.py:45-52`** (`_splitmix64`) — reimplements SplitMix64 PRNG in Python; comment requires it stay "bit-identical to the Rust `intervention_shard::splitmix64`" — a cross-language-duplicated numerical algorithm. **Migration:** expose `intervention_eval_forever_mask(group, seed)` wrapping `InterventionShard::eval_forever_split`; delete the Python splitmix.

- **`intervention_calibration.py:96-225`** (`fit_chart_calibration`) — G3 measurement floor `np.quantile`, per-atom measurability screening, log lifts, per-atom re-speed `s_k=exp((η−mean η)/2)`, held-out RMSE. **Migration:** `intervention_chart_calibration(...) -> PyDict` returning `{respeed, unmeasurable, floor_nats, heldout_rmse, ...}` in a `gam::inference` module.

- **`_fidelity_metrics.py:52-134`** (`loss_recovered`, `r2_score`, `kl_categorical_rows`, `distortion_floor_r2`) — full numpy fallbacks implementing R² (RSS/TSS), categorical KL `Σ p·(logp−logq)`, loss-recovered ratio, and the descending-R² plateau scan, run whenever the Rust `fidelity_*` FFI is absent. A maintained second source of truth. **Migration:** make the existing `fidelity_*` FFI a hard dependency; delete the numpy fallbacks.

- **`_sae_viz.py:260-352`** (`_basis_matrix` + `_periodic_basis`/`_sphere_basis`/`_torus_basis`/`_euclidean_patch_basis`) — reimplements analytic basis evaluation in numpy: sin/cos harmonic design, spherical embedding `(x,y,z,xy,yz,xz)`, tensor-product torus harmonics, quadratic monomial patch — verbatim duplicates of Rust `basis_with_jet`. **Migration:** call `rust_module().basis_with_jet(kind, coords, params)` (already used in `_sae_manifold.py`); delete the four Python builders.

- **`_sae_viz.py:200, 215`** (`_shape_points`/`_token_points`, `return phi @ decoder`) — ambient shape/token point cloud as a Python matmul of basis × decoder. **Migration:** Rust `sae_shape_points_ffi(kind, coords, decoder, params) -> (G,p)` in `gam-pyffi`.

- **`_sae_manifold.py:1292-1319`** (`ManifoldSAE.structure_certificate`) — recomputes e-BH selection statistics in Python: rank by descending `log_e`, threshold `ln(m)−ln(α)−ln(k)`, `e_value=exp(log_e)`, `evidence_remaining_nats=max(0, threshold−log_e)`. The threshold formula is forked from the Rust `_e_benjamini_hochberg`. **Migration:** extend Rust `e_bh_dictionary_certificate` FFI to return per-claim `{rank, threshold, e_value, remaining_nats, confirmed}`.

- **`_sae_manifold.py:4547-4557`** (`_trust_scores`) — per-row/per-atom trust statistics: clip, row-sum normalize (zero-guarded `np.divide`), `per_atom = normalized*atom_trust`, `row = per_atom.sum(1)`. These are the `trust`/`trust_scores` returned by `gamfit.fit()`. **Migration:** `sae_row_trust_scores_ffi(assignments, atom_trust) -> {row, per_atom}` in `gam-sae`.

- **`_sae_manifold.py:1023, 1052`** (`from_payload._periodic_shape_band`) — argsorts the coord grid and computes posterior shape-band mean `mean = phi @ decoder` in Python (surfaced by `shape_uncertainty`). **Migration:** have the Rust shape-band producer emit `shape_band_mean` directly (it already emits `shape_band_sd`).

- **`_sae_manifold.py:3612-3618`** (`StagewiseSAE.reconstruction_ev`) — explained variance `1 − Σ(x−recon)²/Σ(x−x̄)²`, becomes `reconstruction_r2`. **Migration:** Rust `centered_explained_variance_ffi(x, recon)` or return EV from the stagewise payload.

- **`_linear_dictionary.py:54-57`** (`LinearDictionaryFit.reconstruct`) — `recon = codes @ self.atoms + self.mean` in Python; the sibling `transform` already routes through `linear_dictionary_transform_ffi`, so reconstruct is the asymmetric gap. **Migration:** `linear_dictionary_reconstruct_ffi(codes, atoms, mean, centered)`.

- **`_select_topology.py:491-609`** (`stack_topologies` + `_holdout_predictive_moments` 590-609 + `_gaussian_logpdf` 581-587) — assembles the held-out Gaussian log-score objective matrix: recovers per-point σ by inverting the prediction interval `sd=(hi−lo)/(2z)`, computes normal quantile `z=inv_cdf(0.5+0.5·level)`, fills `log_density_rows[i][k]` with `-0.5·log(2π)−log(sd)−0.5·z²`. Only the final weight solve is in Rust. **Migration:** extend `stacking_weights_from_log_density` (or add `stack_topologies_gaussian(names, y, means, sds, level)`) to build the log-density matrix + σ-recovery + quantile in Rust.

- **`_model.py:1451-1456`** (`MultinomialModel.summary`) — per-class coefficient L2 norm `‖β_a‖₂=sqrt(Σ c²)` by slicing the flat coef vector `coefs[a::m]` and hand-rolling `math.sqrt(sum(c*c))`. Display-only but genuine math on model coefficients. **Migration:** `multinomial_model_metadata_pyfunc` returns `slope_norm_per_class: Vec<f64>`.

- **`_basis_eval.py:429-434, 513-518`** (dispatch at `smooth.py:389-395`) — joint tensor-product B-spline design as a row-wise Khatri-Rao/Hadamard-outer product of per-marginal bases (`(out[:,:,None]*col[:,None,:]).reshape`). Numpy path has no torch involvement; the Rust engine already owns a `tensor_bspline` smooth kind. **Migration:** `tensor_bspline_basis(coords, marginals) -> f64[N,ΠKᵢ]` (+ jet variant) in `crates/gam-pyffi/src/`.

- **`torch/interventions.py:310-379`** (`harmonic_code_features`, `synthesize_measure_code`, `spike_edit_code_delta`, `spike_edit_delta_x`) — basis eval `u(t)=[cos2πht, sin2πht]`, measure synthesis `z=Σ a_j u(t_j)`, code→p-space lift `Δx=Δz·D` in numpy; docstrings state these duplicate `gam_sae::sparse_dict::coordinate` and `gam_sae::super_resolution`. **Interop defense: FAILS** (pure numpy). **Migration:** `sae_harmonic_synthesize` / `sae_spike_edit_delta_x(decoder, spikes, edit)` reusing the existing `super_resolution`/`sparse_dict` code.

- **`torch/harvest.py:207-284`** — `_top_r_eigenpairs` (randomized subspace iteration + Rayleigh–Ritz `eigh`), `_trace_estimate` (exact-basis / Rademacher Hutchinson), `_orthonormalize` (QR). *(Listed CORE-MATH by content but see BORDERLINE note — inseparable from the torch-autograd matvec; genuine gray area.)*

## MODEL-LOGIC violations

- **`torch/manifold_sae.py:1096-1317`** (`_SparsityLayer.reconstruction_topk_gate`) — a full deterministic-annealing EM router: per-atom NNLS codes `(recon·x)/‖recon‖²`, squared/relative residuals, `softmax(−resid/τ)` responsibilities, soft→hard STE interpolation across the τ schedule, anchor selection, commitment blending. **Interop defense: FAILS** — not tape plumbing around a Rust call; a full routing/assignment solver in torch, exactly the class the 2026-07-08 coordinate-projection-solver deletion targeted; the closed-form lane already solves the identical problem in Rust (`reseed_atoms_onto_distinct_residual_pcs`, residual-EM). **Migration:** port the residual-EM/anneal core to `crates/gam-sae/`; FFI `sae_reconstruction_topk_gate(x, per_atom_recon, tau, target_k, step, commit_steps, ema_state) -> (gate, assignments)`; needs a custom `autograd.Function` with a Rust backward (VJP w.r.t. `per_atom_recon`).

- **`torch/manifold_sae.py:947-1068`** (`_quadratic_subspace_anchor`) — union-of-subspaces model-selection search: loops all `(i,j)` signed cross-terms, tests median/zero thresholds, per-cluster PCA tail-energy residuals via `torch.linalg.svdvals`, ranks by cross-feature margin, accepts on confidence thresholds. **Interop defense: FAILS** (`no_grad`, pure geometry/statistics). **Migration:** `gam-sae` clustering module; FFI `sae_quadratic_subspace_anchor(x, anchor_subspace_dim) -> (onehot, rule_ij_threshold, confident)`.

- **`torch/manifold_sae.py:1319-1431`** (`_matching_pursuit_commit`) — residual-PC commitment solver: SVD of atom-0's residual, sign-of-top-PC split, per-atom reconstruction-residual argmin routing, median-gap balance fallback. **Interop defense: FAILS** (no_grad geometry; explicitly the "gradient-path analogue of the closed-form lane's residual-energy assignment"). **Migration:** `sae_matching_pursuit_commit(x, per_atom_recon, code, step, commit_steps) -> onehot` in `gam-sae`.

- **`_select_topology.py:226-422`** (`select_topology` + helpers `_screen_candidates_with_budget_cascade` 144-179, `_score_for_kind` 879-900, `_tk_score_from_parts` 910-935, `_scale_score` 956-973) — the model-selection algorithm in Python: escalating budget-cap screening cascade (iterative survivor admission), score arithmetic combining raw REML with the Tierney-Kadane normalizer / `n_obs` / `effective_dim`, survivor sort, winner selection. **Migration:** push score-composition + scaling + cascade decision into `gam-pyffi` alongside `rank_topology_candidates`; add `score_topology_candidate(fit_summary_json, kind, n_obs, basis_size, null_dim, score_scale) -> f64` + a cascade driver.

- **`_select_topology.py:460-488`** (`TopologyStack.predict`) — stacked predictive mixture `Σ_k w_k·μ_k(x)` (weighted accumulation). **Migration:** `stacked_mixture_mean(weights, per_candidate_means) -> Vec<f64>`.

- **`_survival.py:211-243, 458-527`** (helpers 197-209, 245-250) — decides parametric-exponential closed-form hazard vs. finite-differencing a saved cumulative-hazard surface, and implements the tile-stitched forward-difference algorithm (carrying `previous_cumulative`/`previous_time` across tiles, resetting on row-key change). **Migration:** the atomic numerics already exist in Rust (`survival_block_hazard`, `hazard_from_cumulative`); fold surface-selection + tile-stitching into one `survival_hazard_at(surfaces, times, chunk_cfg)` FFI.

- **`torch/harvest_contract.py:64-148`** (`_stable_uniform` + `select_importance_subsample`) — Efraimidis–Spirakis weighted-reservoir sampling without replacement: BLAKE2b-keyed deterministic uniforms, per-row keys `u**(1/w)`, top-k with tie-break. **Interop defense: FAILS** (pure numpy/hashlib, no torch). **Migration:** `sae_select_importance_subsample(importance, n_select, seed, row_keys) -> indices` in `gam-sae`/sampling core.

- **`torch/fit.py:339-366`** (Pca branch) — PCA projection basis via `torch.linalg.svd` of centered points, slice K, `design = points @ basis`, identity ridge penalty. **Migration:** reuse the Rust PCA basis builder behind `gam-pyffi` (`pca_basis(points, K, centered) -> basis`).

- **`torch/fit.py:446-486`** (Categorical branch) — drop-last sum-to-zero contrast coding: one-hot scatter, `onehot[:,:c]-onehot[:,c:c+1]`, identity penalty. Level codes are structural (no gradient). **Migration:** `categorical_contrast_design(levels, n_levels) -> (design, penalty)` in the terms crate.

- **`torch/fit.py:535-632`** (`_shape_constraint_grid_1d`, `_build_shape_constraint_inequality`) — reconstructs the shape-constraint grid (`clamp(unique,96,320)`) and assembles finite-difference inequality rows for monotone (`B[i+1]−B[i]`) and convex (`B[i+2]−2B[i+1]+B[i]`) constraints, culls near-zero rows. `b_grid` is `.detach()`ed — no autograd. **Migration:** `shape_constraint_inequality(smooth_spec, points, kind) -> (A, b)` in `gam-pyffi`.

- **`torch/modules.py:410-427`** (`SparsityPenalty.forward`) — L1 mean, L0 soft `1−exp(−|z|/ε)` STE, Hoyer `((l1/l2−1)/(√F−1))`. **Migration:** add `l1`/`l0`/`hoyer` descriptors to the analytic penalty registry and route via `_RustPenaltyFn`.

- **`torch/modules.py:460-469`** (`SoftmaxAssignmentSparsityPenalty.forward`) — tempered softmax, descending sort, tail-mass beyond `target_k`. **Migration:** `softmax_topk_tailmass` descriptor in the sparsity penalty registry.

- **`torch/modules.py:45-84`** (`_hard_topk_mask`, `_masked_ste`, `TopKActivationPenalty.forward`) — hard top-k boolean mask (topk + scatter) with STE. **Migration:** fold into the topk activation kernel; keep only the STE tape trick in torch.

- **`distill.py:362-418`** (`encode_with_fallback`) — acceptance-gate refinement rule: per-row `assign_err`/`coord_err` L∞ vs a cold exact solve, `accepted=(assign_err≤tol)&(coord_err≤tol)`, rowwise select fast vs exact, fallback-rate stats. **Interop defense: PARTIAL/FAILS** — encoder forward is torch, but the accept/fallback decision + error reductions are numpy control flow deciding per-row output. **Migration:** `sae_encode_acceptance_gate(fast_assign, exact_assign, fast_coords, exact_coords, assign_tol, coord_tol) -> (gated, stats)`.

- **`intervention_calibration.py:55-61`** (`_eval_forever_mask`) — G2 train/eval split rule from hashed `(group ⊕ seed_mix) & 1`, decides which rows enter the fit vs held-out report. **Migration:** same `intervention_eval_forever_mask` FFI as `_splitmix64`.

- **`_sae_manifold.py:263-269`** (`_active_threshold_for_assignment`) — per-assignment-kind active-atom cutoff policy (`np.nextafter(1/K,+inf)` softmax, `finfo.tiny` threshold_gate, `1e-8` ibp_map); drives `summary`, `per_atom_active_set`, `description_length`. **Migration:** `sae_active_threshold(kind, k_atoms) -> f64` in `gam-sae`.

- **`_sae_manifold.py:346-351`** (`_canonical_n_harmonics`; duplicated at 1034-1036, 3498-3501, 3653) — infers periodic harmonic count `H=max(1,(M−1)//2)` from decoder width; affects OOS reconstruct/steer basis width. **Migration:** `periodic_harmonics_from_width(width) -> usize` reused at all four call sites (currently a drift hazard).

- **`_sae_manifold.py:4525`** (`_default_research_k`) — heuristic atom-count selection `max(1, min(N−1, 8, max(2, ⌊√N⌋)))` used by `gamfit.fit()`. **Migration:** `sae_default_research_k(n_obs) -> usize`.

- **`torch/manifold_sae.py:1070-1094`** (`_apply_global_anchor_rule`) — applies cached `(i,j,threshold)` split to a new batch (row-normalize, cross-term product, threshold to one-hot). **Migration:** fold into the `gam-sae` anchor FFI (`sae_apply_anchor_rule(x, rule)`).

- **`sklearn.py:229-243`** (`GAMClassifier.fit`) — `np.unique` derives `classes_`, two-class guard, positive-class convention `positive=classes[1]`, label recode `np.where(labels==positive,1,0)`. **Interop defense: partial** (sklearn interop) but the positive-class rule + {0,1} recode is model semantics duplicating the fit-time schema. **Migration:** `binary_label_encode(labels) -> (classes, encoded, positive)`.

- **`sklearn.py:487-494`** (`GAMClassifier.score`) — restricts AUC to strictly-positive `sample_weight` rows as the weighting semantics for rank-based AUC. **Migration:** `auc_from_predictions_weighted(obs, p, w)` owns the weight-filtering convention.

- **`sklearn.py:415`** (`GAMClassifier.predict`) — class decision `classes_.take(argmax(predict_proba,1))`. **Migration:** return argmax/index from the Rust probability FFI, or `binary_decision(p1)`.

## BORDERLINE (marshalling-adjacent)

- **`torch/harvest.py:186-204, 207-284, 482-500`** — the eigensolver/trace/QR (`_top_r_eigenpairs`, `_trace_estimate`, `_orthonormalize`) are reusable numerical LA that gam owns in Rust (faer), **but** they are inseparable from the torch-autograd `_pullback_matvec` (JVP→Fisher-apply→VJP through the *user's* model), which is genuine interop and qualifies. This is the one place where "written in torch but not cleanly a Rust kernel" holds; a Rust port would need a Rust-calls-into-torch callback FFI. Lower priority — genuine gray area.
- **`torch/manifold_sae.py:1433-1459`** (`_update_assign_ema`) — persistent per-row routing accumulator `β·ema+(1−β)·signal`; model-behavior state (reported assignment is its argmax). Migrate with the routing solver.
- **`torch/manifold_sae.py:834-881`** (`_topk_gate`/`_topk_mask`) — hard top-k selection composing `act*mask` with masked-gradient semantics; `_topk_activation` correctly routes to Rust. Could fold into a Rust `topk_gate`.
- **`torch/interventions.py:135-141`** (`_kl_from_logits`) — `KL(softmax(clean)‖softmax(patched))` in nats; operates at the model-interaction boundary (sanctioned harvest surface). Defensible as interop measurement.
- **`torch/skip_transcoder.py:228-263`** (`attribution_edges`) — `contrib = mean_b z · W_dec[:,j]` + top-k interpretability diagnostic on a trainable torch module; heavy REML/selection already routes to Rust.
- **`torch/harvest_contract.py:208-246`** (`row_metric`/`metric_quadratic_form`) — `W_n=U Uᵀ`, `‖Uᵀv‖²`; trivial numpy LA, consumer-side graceful-degradation contract.
- **`torch/modules.py:336-381`** (`GatedSAEDecoder`) — gated decoder forward with learnable `nn.Parameter`s; standard differentiable layer algebra (genuine torch-module territory). Only the gate STE mirrors the Rust jumprelu gate.
- **`torch/distributed_reduce.py:110-254`** (`build_reduction_tree`, `TreeReducer._reduce_subtree`) — hand-rolled numpy binary reduction-tree fold-order for bit-reproducible cross-node reduction; **not** `torch.distributed` plumbing — a Python re-implementation of the intra-node Rust pairwise-tree discipline. Mild; fold-order arguably belongs beside the Rust reducer.
- **`_api.py:2618-2630`** (`_resolve_position_basis_inputs`, `eff_period`) — periodic-Duchon wrap period from knot geometry `eff_period=span+mean_spacing`; documented correctness role (undersized period → non-PSD Gram, gam#580). Leans CORE. **Migration:** let the Rust periodic-Duchon builder auto-derive the wrap period.
- **`_api.py:2660-2679`** (`_resolve_periodic_position_bspline_knots`) — periodic B-spline knot placement `np.linspace(origin, origin+period, k+1)`, contradicting the module's own note that placement is Rust-owned. **Migration:** `auto_periodic_bspline_knots(t, k, period)` FFI.
- **`_api.py:598-630`** (`_normalize_fisher_rao_w`) — embeds scalar/vector into `(N,d,d)` diagonal precision blocks + symmetry/non-negativity validation. Mostly marshalling; the diagonal-embedding is a small structured-matrix transform.
- **`_model.py:1293-1298`** (`MultinomialModel.predict`) and **`_select_topology.py:552`** — two-sided normal quantile `z=inv_cdf(0.5+level/2)` before passing `z` to Rust. **Migration:** pass `level` and take the quantile in Rust.
- **`_model.py:908-994`** (`Model.partial_dependence`) — grid construction with midpoint `0.5*(lo+hi)` + `np.linspace`; the estimand + delta-method SE are Rust. Grid-building for FFI input.
- **`_model.py:1059-1062`** (`Model.bayes_factor_vs`) — `math.exp(log_diff)` on a Rust-provided log Bayes factor. Nearly pure marshalling.
- **`_select_topology.py:1055-1062`** (`_effective_dim_value`) — fallback `float(sum(value))` totals a per-term EDF vector. Minor aggregation.
- **`_sae_manifold.py:290-302, 314-326`** (`_channel_cov_factors`/`_channel_cov_from_factors`) — reshapes `(M·p, M·p)` decoder covariance to `(M,p,M,p)`, extracts same-channel diagonal, rebuilds block-diagonal on load. Save/load restructuring feeding the shape band.
- **`_sae_manifold.py:1682-1697`** (`build_encode_atlas` defaults) — default `amplitude_bounds` as per-column `|assignments|.max()` and `target_norm_bound` as `max_i ‖x_i‖₂`. Simple reductions producing Rust inputs. **Migration:** let Rust `build_sae_encode_atlas` derive defaults.
- **`_sae_manifold.py:2137-2139`** (`description_length`) and **`_linear_dictionary.py:114`** — `coord_dim=np.mean(coord_dims)`, `n_params=Σ b.size`, training column mean for the centered K=1 lane. Scalar/vector reductions feeding a Rust core.
- **`identifiability.py:603-758`** (`_one_fit`) + **761-855** (`identifiable_factor_fit`) — torch encoder/decoder training loop: RSS objective, Adam steps, Rust-analytic penalty grads injected via surrogate losses. **Interop defense: LARGELY HOLDS** — the encoder is a torch `nn.Module` explicitly not in the Rust REML engine; every statistical primitive already routes to Rust. Residual Python math (RSS reduction + surrogate chain-rule wiring) is intrinsic to driving torch autograd.
- **`_basis_eval.py:388-391, 536-538`** (`pca_evaluate`/`pca_evaluate_numpy`) — mean-centering + `Φ(x)=(x−mean)·basis`. Numpy path undefended; one centered gemm on a user basis; Rust engine exposes a `pca` smooth kind. **Migration:** `pca_project(coords, basis, centered)`.
- **`smooth.py:699-716`** (`Sphere.basis_size`) — closed-form basis-dimension identity `L·(L+2)` (harmonic) / `n_centers−1` (Wahba); duplicates the Rust builder's dimension law (drift risk). **Migration:** `sphere_basis_size(n_centers, kernel, centers)`.
- **`smooth.py:821, 831-832`** (`PeriodicSplineCurve._evaluate_torch/_numpy`) — periodic coord wrap `t←t−floor(t)` before the Rust cyclic-basis call. **Migration:** fold the mod-1 reduction into the Rust kernel.
- **`distill.py:73-77, 263-266, 330-336`** (`_scale` + standardization + tolerance calibration) — `(x−mean)/std` with zero-std guard + acceptance-tolerance calibration `coord_tol=max(max(coord_err[cal])·mult, 1e-10)`. The Adam/MLP loop is genuine torch; the numpy standardization + threshold-calibration set model-behavior constants. Fold into the acceptance-gate/teacher-solve FFI.
- **`crosscoder.py:320-358`** (`_decoder_column_norms`, `atom_layer_affinity`, `harmonic_atoms`) — decoder column L2 norms (read from torch weights), row-normalized affinity, `affinity≥tol` mask. The training loop is torch-exempt; the numpy normalization/thresholding could be a Rust reduction. **Migration:** `crosscoder_atom_layer_affinity(col_norms, tol) -> (affinity, harmonic_idx)`.
- **`_manifold.py:542-553`** (`Poincare.metric`) — conformal metric `g=λ²·I` in numpy after fetching scalar `λ` from Rust; every other Poincaré primitive delegates per-point. **Migration:** `poincare_metric_tensor(point, curvature)`.
- **`_diagnostics.py:169`** and **`sklearn.py:447`** — null base rate `train_prev=float(np.mean(observed))` computed in Python to feed Rust `classification_metrics` (for `nagelkerke_r2`). **Migration:** drop the parameter; let Rust derive the base rate.
- **`sklearn.py:352-354`** (`GAMClassifier.predict_proba`) — `positive=np.clip(mean,0,1)`, `negative=1−positive`, column_stack. `_predict_shape.py` routes the same clip through Rust (`marginal_slope_clip_probabilities`) — inconsistency. **Migration:** `binary_class_probabilities(mean) -> (N,2)`.
- **`diagnostics/anchor_consistency.py:124-156`, `aux_richness.py:105-166`, `jacobian_sparsity.py:140-203`** — pass/fail verdicts + derived ratios on top of Rust kernels (`n_anchors>=K`, `rank>=latent_dim`, `mean_sparsity>=threshold`, `anchor_fraction=n_anchors/n`, `min(ranks)`, `(N,P,K)→(N·P,K)` reshape at `jacobian_sparsity.py:142`). **Migration:** have the `diagnostics_*` pyfunctions return the booleans + fractions and accept the raw `(N,P,K)` stack.
- **`_basis_protocol.py:334-403`** (`BasisDescriptor.jacobian`/`.hessian`) + **259-290** (JAX JVP `jnp.einsum("bmk,bk->bm", jac, xdot)`) — per-column `torch.autograd.grad` loops building `(B,M,d)`/`(B,M,d,d)`, and the JAX custom_jvp contraction. **Interop defense: LARGELY HOLDS** (frontend autograd; code notes it is a stopgap "until the Rust value_grad FFI lands"). **Migration:** replace with `basis_value_grad`/`basis_value_hess` bridges when they land.
- **`_protocol.py:173-187`** (`PenaltyDescriptor.hessian_diag`) — per-coordinate `hvp(t, e_i)` probe loop. **Interop defense: mostly holds** (torch tensors via torch `hvp`), but the `e_i`-probe diagonal assembly is generic LA the Rust penalty registry could expose. **Migration:** `analytic_penalty_hessian_diag`.

---

## Clean modules (verified thin wrappers, no violation)

`_reml_common.py`, `_reml.py` (all wrappers route forward+backward through Rust adjoints), `hyperbolic.py` (every Poincaré/Lorentz op routes to `gam::geometry::poincare`), `_penalty_jax_vjp.py` (JAX `custom_vjp` shell around the Rust callback — the sanctioned external-software pattern), `_penalties.py`/`_penalty_bridge.py`/`_penalty_descriptors.py`/`_penalty_frames.py`/`_composite_penalty.py` (value/grad/hvp via `analytic_penalty_*`), `_response_geometry.py` (all charts via `response_geometry_*`; the SPEC-violating ALR `eigh` whitener was already removed), `torch/interchange.py` (pure autograd shims wrapping Rust fwd + Rust analytic bwd — the sanctioned pattern), `manifold_sae.py` `_BasisWithJetFn`/`_sinkhorn_balance`/`_project_to_manifold`, `_smooth.py`, `_predict_shape.py`, `_summary.py`, `_tables.py`, `_basis_descriptors.py`, `_cuda.py`, `_binding.py`, `_warnings.py`, `_compare.py`, `_exceptions.py`, `_schema.py`, `_validation.py`, `_frame*.py`, `_diagnose_plot.py`, `diagnostics/_report.py`, `diagnostics/__init__.py`, `_sae_trust.py`, `_sae_spectral.py`, `_dispatch.py`, `_coerce.py`, `_torch_compat.py`, `module.py`.

---

## Parallelization plan for migrations

Migrations are naturally partitioned by **Python source file** (each file below is touched by exactly one cluster, so no source-file write conflicts) but several target the **same Rust crate module** — that is the real serialization constraint. Group so agents never touch the same Rust file:

**Fully independent (distinct Python file AND distinct Rust target — safe to run in parallel):**
- **A. Topology** — `_select_topology.py` → `gam-pyffi` topology entries + a topology score/stacking Rust module.
- **B. Survival** — `_survival.py` → survival crate + `survival_hazard_at` FFI.
- **C. Compositional + sphere geometry** — `torch/geometry.py` → response-geometry / `gam::geometry` crate (distinct from the Poincaré-owned hyperbolic path).
- **D. Ridge VJP** — `torch/_basis.py` (`_gwr_vjp`) → gam solve/linalg crate `gaussian_weighted_ridge_backward`.
- **E. Fidelity metrics** — `_fidelity_metrics.py` → delete numpy fallbacks; `fidelity_*` FFI already exists (pure Python-side deletion, zero Rust conflict).
- **F. sklearn + diagnostics** — `sklearn.py`, `_diagnostics.py`, `diagnostics/*.py` → classification-metrics + `diagnostics_*` pyfunctions (return verdicts/fractions).
- **G. Basis/terms design** — `torch/fit.py` (tensor-bspline/pca/categorical/shape-constraint) + `_basis_eval.py`/`smooth.py` tensor-product — basis/terms crate. (Keep as one owner since both touch tensor-bspline lowering.)

**Serialize within — all target `crates/gam-sae/` assignment/routing/penalty modules (partition by Rust submodule to parallelize safely):**
- **H1. Routing solvers** — `torch/manifold_sae.py` (reconstruction_topk_gate, quadratic_subspace_anchor, matching_pursuit_commit, direction_cluster_anchor, apply_global_anchor_rule, update_assign_ema, topk_gate) → new `gam-sae` routing module. Single owner (one Python file, one new Rust module).
- **H2. SAE activation/penalty kernels** — `torch/penalties.py` + `torch/modules.py` → `gam-sae` `assignment.rs`/`sparsity.rs` + analytic-penalty registry. Conflicts with H1 only if they share `assignment.rs`; give H2 the penalty-registry + activation-kernel files and H1 the new routing module → parallel-safe.
- **H3. Encode-assignment + interventions** — `distill.py`, `intervention_calibration.py`, `torch/interventions.py`, `torch/harvest_contract.py` → `gam-sae` encode/`gam::inference` + sampling. Distinct Rust files from H1/H2 → parallel-safe.
- **H4. Manifold-SAE payload math** — `_sae_manifold.py`, `_sae_viz.py`, `_linear_dictionary.py`, `crosscoder.py`, `_manifold.py` → `gam-sae` shape-band/trust/e-BH/basis-eval + `gam::geometry::poincare`. Mostly new FFI surfaces; distinct from H1-H3.

Recommended first wave (highest value, zero mutual conflict): A, C, D, E, F, H1. Second wave: B, G, H2, H3, H4.

# Manifold Crosscoder — Design (promote multiblock M1 into the unified engine)

Status: DESIGNED 2026-07-09 (read-only design run). Implementation in increments
A–E below; each keeps the workspace green. Companion to
docs/design/sae-unification.md — the crosscoder must be a SCHEDULE of the one
engine, never a sixth fit path.

## 0. Executive summary and the surprise in the tree

The crosscoder is NOT greenfield. An M1 primitive already landed and is green on main:

- `SaeManifoldTerm::run_multiblock_reml_fit` — crates/gam-sae/src/manifold/behavior_fit.rs:162 — one shared latent `t` + gates decoded through an AUGMENTED STACKED TARGET `Z-tilde = [Z | sqrt(lambda_1)*Y_1 | ... | sqrt(lambda_{L-1})*Y_{L-1}]`, per-block relevance lambda_l REML-selected by the closed-form variance ratio, Armijo-safeguarded against the block-abandonment runaway (behavior_fit.rs:213-218; criterion `profiled_reml_criterion` at behavior.rs:910).
- `OutputBlock` (behavior.rs:726) with `split_honest_decoder` (behavior.rs:794) and `stack_augmented_target` (behavior.rs:849).
- Tests: manifold/tests_crosscoder_multiblock.rs (bit-identical K=2 reduction to the two-block driver at :207; 2-layer and 3-layer synthetic crosscoders with lambda ordered by planted noise at :304 and :390), plus a real-data CLI example crates/gam-sae/examples/curved_crosscoder.rs (row-aligned per-layer .npy + per-layer PCA; the "development-coder" checkpoint-axis follow-up is documented in its module docs :20-27).

What the epic actually is: promote this standalone FIXED-rho driver into the unified engine (docs/design/sae-unification.md, ratified 2026-07-09) — per-block weights as rho coordinates of the outer REML objective, admission through the front door, support-sparse K>P operation, pyffi entry, OLMo multi-layer evaluation, and the cross-layer drift statistic. The multiblock driver must end as a SCHEDULE of the one engine, exactly as fit_tiered does in the unification plan — not a sixth fit path.

---

## 1. Q1 — State shape: stacked (N, L·P), one term, block-columned decoders

**Decision: the target stays a single stacked (N, p~) matrix, p~ = sum_l p_l; ONE SaeManifoldTerm; each atom keeps ONE `decoder_coefficients: (M_k, p~)` (atom.rs:502-511) whose column blocks are the per-layer decoders B_k^(l).** This is what M1 already does: run_multiblock_reml_fit validates `p_tot == self.output_dim()` at behavior_fit.rs:196-203 ("atoms must be built at the augmented width").

Why NOT L atoms sharing coords:
- Routing is per-atom. L atoms per feature would let the router activate layer-l's copy without layer-(l+1)'s — destroying the shared-code constraint that IS the crosscoder. Coord-tying across atoms has no representation in SaeAssignment/SaeAssignmentState (coords are per-(row, atom): assignment_state.rs:80-84) and would need new tying machinery through every Newton block.
- Basis evaluation Phi(t_i) would be recomputed L times for identical t_i.

Why stacking keeps the arrow-Schur border math unchanged:
- The decoder border width is `beta_dim() = sum_k M_k * p` (construction.rs:2292-2295) — stacking layers IS just p -> p~. The border LSQ, htbeta, and penalty ops are already generic in p; no assembly change.
- The FRAMED border is the scaling escape hatch: `factored_border_dim() = sum_k M_k * r_k` (construction.rs:2313-2321) is INDEPENDENT of p~ once Grassmann frames are active (B_k = C_k U_k^T, U_k in Gr(r_k, p~)). The border Cholesky / evidence log-det scale with the factored count (in-source comment construction.rs:2317-2318), so the L-times widening lands on the cheap frame side, not the (border)^2 Schur side. The dense full-B border Hessian is (sum M_k * p~)^2 — the one thing quadratic in L, which is why frames should be the DEFAULT in the crosscoder lane (risk §5.1).
- Per-row blocks (t_i, gates) do not see p~ at all; only the row-residual accumulation widens (linear in p~).

Per-layer decoder access is a column-slice view + honest unscaling, exactly as the test does today (tests_crosscoder_multiblock.rs:377-383): `B_k^(l) = decoder_coefficients[:, off_l..off_l+p_l] / sqrt(lambda_l)`. Add a first-class accessor `SaeManifoldTerm::layer_decoder(k, l)` carried by a new `CrosscoderLayout { p_x, block_dims: Vec<usize>, labels: Vec<String> }` stored on the term (next to the behavior block; SaeManifoldTerm at term.rs:377), so no caller recomputes offsets by hand.

**Assignment is untouched.** SaeAssignment stores (logits (N,K), coords, mode, ungated, frozen_logits) and SaeAssignmentState stores (indices, gate_params, coords) (assignment_state.rs:72-93) — verified neither has any P dependence. Routing is layer-shared by construction because gates multiply the whole row of the augmented reconstruction.

---

## 2. Q2 — Penalty/REML structure

### 2a. Per-layer relevance lambda_l as rho coordinates (the unification move)

Today run_multiblock_reml_fit alternates (fixed-rho inner joint fit) <-> (closed-form lambda update) with NO smoothness/ARD selection in the loop — rho is a caller-passed constant (behavior_fit.rs:166). The unified design makes log lambda_l outer coordinates.

Extend SaeManifoldRho (rho.rs:36-62) with:

    pub log_lambda_block: Vec<f64>,   // length L-1, block order; EMPTY for a plain SAE

Flat layout becomes `[log_lambda_sparse, <K smooth>, <ARD>, <L-1 block>]` — APPENDED, so every existing consumer's cursor arithmetic (to_flat/from_flat at rho.rs:365/423, ard_flat_index at rho.rs:152) is untouched when the vector is empty; the plain-SAE path is byte-identical.

**The criterion.** The lambda_l-dependence is exactly the already-derived profiled form (behavior.rs:890-917):

    C(rho) ⊇ (n*p~/2) * log( (R_x + sum_l lambda_l*R_l) / (n*p~) )  −  sum_l (n*p_l/2) * log lambda_l

with unscaled per-block residuals read off the fitted state by the existing `augmented_block_rss` (behavior_fit.rs:442). The second term is the sqrt(lambda_l) target-scaling Jacobian — it makes lambda->0 pay +inf and prevents the block-abandonment runaway.

**Analytic gradient** (new coordinate, no FD — mandatory under the FD-ban gate):

    dC/d(log lambda_l) = (n*p~/2) * lambda_l*R_l / (R_x + sum_m lambda_m*R_m)  −  n*p_l/2

(envelope theorem at the inner optimum: dR/dbeta * dbeta/dlambda terms vanish, the same argument the landed lambda-gradient channels use). Its zero is `lambda_l = (R_x/p_x)/(R_l/p_l)` — the landed closed form (behavior.rs:703, OutputBlock::reml_updated_log_lambda at behavior.rs:811) becomes the EFS-style fixed-point update for this coordinate inside `efs_step` (outer_objective.rs:3343), and the analytic gradient feeds the quasi-Newton lane. This literally re-derives M1's alternation as one more Fellner–Schall coordinate — the driver's semantics are preserved, its loop body is deleted.

**Mechanics of a rho-dependent target.** SaeManifoldOuterObjective.target (outer_objective.rs:698) is currently a fixed matrix. Do NOT re-stack per eval: when block coordinates move from lambda_old to lambda_new, rescale that block's column range IN PLACE by sqrt(lambda_new/lambda_old) at the top of `fit_at_fixed_rho` (outer_objective.rs:3018). O(N*p_l) per moved coordinate, allocation-free, and every downstream lane (basin bundle at outer_objective.rs:758-775, row subsample at :740, streaming criterion, surrogate lane at :757) reads the coherently-scaled target with no further change. Rescale from ONE pristine unscaled copy of the block columns (kept once) rather than multiplicatively, to avoid drift over thousands of evals.

### 2b. Does the evidence factor stay block-diagonal per layer? Yes for lambda_l — exact statement

Given shared (t, gates), the Gaussian decoder normal equations are COLUMN-SEPARABLE with one shared Gram: the data-fit Hessian over decoder columns is I_{p~} ⊗ G (G = gated basis Gram) and today's smoothness penalty is column-uniform, lambda_k^smooth * (I_{p~} ⊗ S_k). Hence the decoder-block evidence log-det factorizes:

    log|H_bb| = p~ * log|G + sum_k lambda_k S_k ⊕ ...|    (one M-space factorization, weight p~)

Scaling target columns by sqrt(lambda_l) does not touch the DESIGN, so lambda_l never enters log|H_bb| — it enters only through RSS and the Jacobian term. The evidence stays exactly as cheap as the single-layer fit. This is the deep reason the closed-form lambda_l update decouples per block (the cancellation argument documented at behavior.rs:700-712).

### 2c. Per-layer smoothness — the hidden coupling, and the honest fix

The smoothness penalty acts on the SCALED decoder B~^(l) = sqrt(lambda_l) * B^(l), so in honest units layer l's effective smoothness is lambda_smooth * lambda_l — REML down-weighting a noisy layer also RELAXES its smoothness. Directionally right, but a tied choice, not a selected one.

If decoupling is warranted (decide by measurement — Inc E), the extension is a per-layer smoothness multiplier mu^(l): penalty sum_k lambda_k sum_l mu^(l) tr(B~_k^(l)T S_k B~_k^(l)). Column-uniformity breaks per layer block only, so the evidence log-det becomes a SUM OF L CHEAP M-SPACE FACTORIZATIONS:

    log|H_bb| = sum_l p_l * log|G + sum_k lambda_k mu^(l) S_k ⊕ ...|

— still block-diagonal per layer given shared t; L-1 more rho coordinates (anchor's mu pinned to 1 for identifiability against lambda_smooth). The ArdSharing::Shared machinery (rho.rs:27-32) is the template for keeping outer coordinate counts constant in K.

---

## 3. Q3 — What the shared-coordinate constraint buys

1. **Cross-layer feature identity by construction.** One (t_i, gate_i) per token explains all layers; "the same feature at layer l and l+1" is not a post-hoc matching problem (the flat-crosscoder literature's Hungarian step) — it is the SAME atom, SAME chart coordinate, different column block.
2. **Parameter count.** L separate SAEs: L*(sum_k M_k*P) decoder + L routing states (N*s*(2+d)*8 bytes each, assignment_state.rs:28-40). Crosscoder: sum_k M_k*p~ = L*(sum_k M_k*P) decoder (same) but ONE routing state and ONE coordinate field — the entire per-row state is shared. Memory win and statistical win (every layer's evidence updates the same t).
3. **The drift statistic.** In the shared chart gauge, feature evolution across depth is
   delta_k(l) = ||B_k^(l+1) − B_k^(l)||_F / sqrt(||B_k^(l)||_F * ||B_k^(l+1)||_F) in honest units (split_honest_decoder).
   Key gauge fact: because all layers decode through the SAME Phi(t), a chart reparameterization t -> phi(t) acts identically on every B^(l) — cross-layer DIFFERENCES are chart-gauge-invariant by construction; only the global SCALE gauge (retract_decoder_gauge_in_loop pins ||B_k||_F = 1 over the full augmented decoder, gauge.rs:600) must be quotiented, which the normalization above does. Also expose per-layer decoder principal angles angle(colspan B_k^(l), colspan B_k^(l+1)) — distinguishes "same feature, rotated readout" from "feature dying", and is lambda_l-independent. The REML log lambda_l per block is itself the layer-relevance readout the M1 tests already assert (tests_crosscoder_multiblock.rs:457-466).
4. **Checkpoint axis for free** — the development-coder: swap layers for training checkpoints; nothing in the driver changes (examples/curved_crosscoder.rs:20-27). Keep block labels semantic-free.

---

## 4. Increment plan (each green, each hand-off-able)

### Inc A — CrosscoderLayout + stacked-target adapter (no new math)
- Files: crates/gam-sae/src/manifold/term.rs (layout field + layer_decoder(k, l) accessor + validation sum p_l == output_dim), behavior.rs (move OutputBlock offset bookkeeping into the layout type), manifold/mod.rs exports.
- Keep bit-parity for the existing fixed-rho callers — the M1 tests are the parity gates.
- Tests: layout round-trip; L=1 (no blocks) byte-identical to the plain fit (extend the K=2 bit-parity pattern at tests_crosscoder_multiblock.rs:207).

### Inc B — log_lambda_block rho coordinates in the outer objective (the unification core)
- Files: rho.rs (field + flat-layout append + from_flat/to_flat), outer_objective.rs (fit_at_fixed_rho:3018 in-place block rescale from pristine columns; criterion adds the −sum (n*p_l/2) log lambda_l Jacobian + pooled-RSS form; efs_step:3343 gains the closed-form variance-ratio coordinate update; analytic gradient per §2a), behavior_fit.rs (run_multiblock_reml_fit becomes a thin schedule over the engine — its alternation loop body deleted, mirroring the unification treatment of update.rs::run).
- Tests: FD-vs-analytic parity on the new coordinates (tests_pen_fd_780.rs pattern); planted 2-layer problem: engine-selected log lambda_l agrees with the M1 closed-form fixed point; port diag_crosscoder_sweep_sensitivity (tests_crosscoder_multiblock.rs:131) as the no-divergence pin.
- Caution: every flat-rho consumer enumerated at rho.rs:126-133 (derivative/trace/EFS/IFT-RHS walkers) must be audited for the appended tail; from_flat length assertions (rho.rs:428) must learn it.

### Inc C — Front door + support-sparse lane at stacked width
- Files: front_door.rs (admit_sae_fit:61, admit_topk_manifold:118). Admission at output_dim = p~ is already CORRECT (the response really is N*p~), but add a border-growth admission term: the dense full-B lane must also check (sum M_k * p~)^2 * 8 against budget, and the crosscoder lane defaults atoms onto frames so the border is factored_border_dim (construction.rs:2319). streaming_plan.rs budget arithmetic gets p~ threaded (framed decoder bytes O(K*M*r) are p~-free).
- Tests: admission-table tests extended with stacked shapes; the "never silently substituted" refusal message (front_door.rs:176-184) names the crosscoder shape.

### Inc D — pyffi entry + OLMo two-layer smoke
- Files: crates/gam-pyffi/src/manifold/geometry_ffi.rs:4813 (extend sae_manifold_fit payload with targets: list[(name, array)]; manifold_sae_coercion.rs payload plumbing), manifold_and_posterior_ffi.rs ManifoldSaePayload (#2091 serde payload) gains block_dims/labels/log_lambda_block so per-layer decoders round-trip.
- Fixture: a row-aligned two-layer OLMo pair — the committed fixtures (olmo_l18_pca64_635.npy, olmo_mixedlayer_pca64_768.npy; resolver at tests_olmo.rs:400-434) are single-layer; commit e.g. olmo_l12_l18_pca32_*.npy (same tokens, per-layer PCA as in the example).
- Tests: smoke — per-layer EV finite, anchor EV competitive with the single-layer fit, log lambda_l finite/identifiable.

### Inc E — Drift statistics + eval harness (+ the §2c per-layer-smoothness decision)
- Files: new manifold/crosscoder_drift.rs (delta_k(l), principal angles, per-layer honest EV), pyffi exposure, saebench_metrics.rs integration (#1942 eval infra).
- Measurement gate for §2c: planted two-layer problem with different per-layer true smoothness; if the tied lambda_smooth*lambda_l measurably mis-smooths the down-weighted layer's honest-units decoder, land the mu^(l) axis (evidence math in §2c); otherwise document the tie as the deliberate default.

---

## 5. Risks

1. **Border quadratic in p~ on the dense full-B path.** beta_dim = sum M_k * p~ and the border Hessian workspace is beta_dim^2 (construction.rs:2297-2306; the K*M*p allocation cost is already flagged in-source at construction_arrow_schur_assembly.rs:579). Mitigation is architectural: crosscoder admission prefers framed atoms (factored_border_dim is p~-independent) and Inc C's admission check refuses shapes whose full-B border can't pay.
2. **Memory N*L*P + per-sweep re-stacking.** stack_augmented_target materializes a fresh (N, p~) per sweep AND per backtracking trial (behavior_fit.rs:225, 260) — O(sweeps * N*L*P) allocations today. Inc B's in-place block rescale removes this class entirely (one pristine + one live copy). Streaming lane: chunked row access of the stacked target is unchanged in shape.
3. **K>P admission semantics shift.** Stacking raises the dense-certification threshold to K <= p~ = L*P — honest by the N*K <= N*P~ rule (front_door.rs:55-77), but it silently admits L-times more atoms to a dense lane whose border is now L-times wider; Inc C's border check is the guard. The TopK CurvedStreaming lane (front_door.rs:88-127) is P-agnostic in its routing state and is the intended large-K crosscoder home.
4. **Routing/gate semantics.** Verified: nothing in SaeAssignment/SaeAssignmentState reads decoder columns (§1). One real seam: SEEDING. PCA/chart seeds and router scores computed from the augmented target must be built at seed weights lambda_l = 1 (or anchor-only) so the seed is not biased by uninitialized weights. Phase-basin conflicts across layers are real — the tests deliberately plant a shared leading phase sense (tests_crosscoder_multiblock.rs:324-330, 408-412); on real data, seed t from the anchor (or joint PCA) and let the basin bundle (outer_objective.rs:758-775) arbitrate. No per-layer phase knob.
5. **Lambda-alternation divergence at small n** is known and already safeguarded (criterion-monotone Armijo, behavior_fit.rs:213-218; regression pin diag_crosscoder_sweep_sensitivity). Folding lambda into rho (Inc B) inherits the outer engine's line search, which subsumes the safeguard — keep the diag test as the pin.
6. **rho-layout blast radius.** Appending block coordinates touches every consumer at rho.rs:126-133; the objective<->gradient desync class (#2087) is the known failure mode — Inc B's simultaneous all-coordinate FD parity test is mandatory, and the analytic gradient must land WITH the coordinate (FD-gated CI), not after it.

# Manifold-SAE / gamfit roadmap (Scott, 2026-06-03)

Engine internals (solver, jets, REML, streaming, topology evaluators) are mature.
Remaining work = correct objective, honest output, parallel hot path, clean API +
diagnostics + viz. Prioritized below. **The three that matter most: (A) incoherence
retargeted to cross-atom decoder subspaces, (B) per-atom trust score + Level-0, (C)
ideal-objective defaults + 4-call API + manifold viz.**

## FOUNDATION (gating): SAE fit must converge with isometry on / multi-atom
Status: circle d=1 (iso off) R²=0.997 ✓. isometry-on → slightly-indefinite Schur
(pivot ~-0.049). Fix tournament in flight (GN cross-term H_tβ / Tikhonov floor /
complete hbb majorizer / LM-escalate log-det). Everything below needs this green.

## ACCURATE (first)
1. Incoherence → cross-atom decoder subspaces, default ON.  STATUS: **mostly done** —
   DecoderIncoherencePenalty (#671) targets cross-atom decoder column-spaces on
   co-activating pairs, default 1.0 for K≥2. block_orthogonality (wrong object) stays
   off. ACTION: audit it matches spec; confirm default-on end-to-end.
2. Per-atom **trust score + diagnostics**, returned by every fit: σ_min(active tangent),
   neighbor coherence, topology evidence margin, coverage/frequency, Level-0 residual.
   THE differentiator. NEW.
3. **Level-0 misspecification test**: nonparametric per-atom reference (local-PCA tangent /
   GP-LVM, no type prior); if typed atom doesn't reconstruct as well → flag **untyped**.
   Feeds trust score. NEW.
4. Wire **nuclear-norm rank penalty** onto decoder block. STATUS: #672 nuclear_norm_weight
   wired — audit it reaches the SAE β payload, confirm default.
5. **Split-sensitive benchmark harness** (permanent infra, not a script): coherence×coverage
   synthetic; scored metrics = per-token coordinate recovery up to isometry (Procrustes) +
   σ_min reporting. NEW.

## FAST
1. Parallelize per-token solve (rayon) — saturate multicore; GPU (A100) offload only if
   tokens are still the wall.
2. Share Jacobian/jet across data-fit/smoothness/isometry (compute once/step).
3. Sparse active-set default at scale (atom_selection bitsets / SaeRowLayout → O(active)).
4. Warm-start per-token solve (prev epoch + cheap init).

## BEAUTIFUL & USEFUL
1. Default config = ideal objective (incoherence-on-decoder, nuclear-on, adaptive sparsity).
2. Clean Python API: `fit(acts,config)→dict(atoms+coords+trust)`, `featurize(acts)→coords`
   (predict_oos, make first-class), `align(a,b)→quotient-aligned`, `plot(atom)`.
3. Reproducibility/alignment as a tool: Hungarian atom-match, Grassmann subspace distance,
   Procrustes coords, topology-flip rate vs evidence margin.
4. Manifold visualization: project atom to subspace, render shape + token coords + topology.
5. Spectral coordinate initializer (Laplacian eigenmap) — init + cheap topology guess.
6. Keep behavioral-metric gauge W exposed as a config flag (isometry penalty already carries it).

## LATER (options, once above lands)
Monotonicity (ordered concepts), nested-prefix/Matryoshka, SCAD/MCP (swap after L1),
cross-layer sheaf consistency (deferred).

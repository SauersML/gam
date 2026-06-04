# Manifold-SAE / gamfit roadmap

Engine internals (solver, jets, REML, streaming, topology evaluators) are mature.
Remaining work = correct objective, honest output, parallel hot path, clean API +
diagnostics + viz. **The three that matter most: (A) incoherence on cross-atom decoder
subspaces [DONE], (B) per-atom trust score + Level-0 misspecification, (C) ideal-objective
defaults + 4-call API + manifold viz.**

## Status
- Convergence: full bisection matrix (line/circle × d∈{1,2} × isometry∈{0,0.1,1.0}) now
  converges; circle d=1 regression fixed (commit "restore full-matrix convergence").
- Open correctness work (see issue #681): scale-free isometry target (G_ref=I misspecifies
  the manifold radius → poor recovery), von-Mises ARD Bessel normalizer, isometry value in
  the REML criterion, joint Gauss-Newton cross-block invariant for the isometry curvature.

## Accurate
1. Incoherence → cross-atom decoder subspaces, default ON (K≥2). DONE (#671).
2. Per-atom trust score + diagnostics every fit: σ_min(active tangent), neighbor coherence,
   topology evidence margin, coverage, Level-0 residual. The differentiator.
3. Level-0 misspecification test: nonparametric per-atom reference; flag `untyped`.
4. Nuclear-norm rank penalty on the decoder block (embedding-dim selection). DONE (#672).
5. Split-sensitive benchmark: per-token coordinate recovery up to isometry (Procrustes) + σ_min.

## Fast
Rayon per-token solve (+GPU only if it's the wall); share jets across data-fit/smoothness/isometry;
sparse active-set default; warm-start per-token solve.

## Beautiful & useful
Default config = ideal objective (incoherence+nuclear+adaptive sparsity); 4-call API
`fit/featurize/align/plot`; reproducibility tool (Hungarian/Grassmann/Procrustes/flip-rate);
manifold visualization; spectral (Laplacian-eigenmap) coordinate initializer; expose behavioral W.

## Later
Monotonicity, nested-prefix/Matryoshka, SCAD/MCP options, cross-layer sheaf consistency.

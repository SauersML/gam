# Issue #1026 — large-K SAE reconstruction parity

## State of the world (from issue thread)
The in-tree ladder is largely landed. Owner reports first real-data EV-vs-K numbers
(OLMo-3-32B L25): curved beats linear at matched K, climb without collapse at K∈{1,2}.

K≥4 blocked by THREE plumbing bugs:
1. K=4 curved: `periodic atom requires positive n_harmonics; got 0` — d_atom=1→n_harmonics
   derivation in Python wrapper yields 0 at K=4. (reported fixed in #1132 bug 1)
2. K=4 linear: `arrow_log_det_from_cache returned None at ridge=0 Direct mode` — Laplace
   normalizer's joint-Hessian log-det not cached in Direct (non-dense-Schur) solver path.
3. K=1 linear held-out: `decoder_blocks[0] has M=1 but rebuilt basis has M=3` — OOS basis-M
   mismatch for euclidean atoms in `sae_manifold_predict_oos`.

## Plan
- Reproduce / locate each of the three bugs in-tree.
- Fix root causes (euclidean OOS M mismatch, euclidean Laplace log-det in Direct mode,
  n_harmonics derivation).
- Add regression tests.
- Verify with ./build.sh nextest.

This is an autonomous WIP, force-improved commit-by-commit.

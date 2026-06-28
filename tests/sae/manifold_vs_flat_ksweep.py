#!/usr/bin/env python3
"""Sparsity (k) sweep: manifold SAE vs flat TopK SAE on a single layer (OLMo l18).

The reconstruction-vs-sparsity Pareto. Fix dict=64, sweep active-k from 1 to 64,
OOS variance explained (2-seed avg). FLAT = dictionary_learning TopK SAE;
MANIFOLD = gamfit.torch.ManifoldSAE (learned encoder, curved Fourier atoms).

RESULT (dict=64, OLMo l18 train 381 / test 254):
     k   MANIFOLD   FLAT    gap
     1     0.143   0.169  -0.026
     2     0.324   0.226  +0.098
     4     0.365   0.257  +0.108
     8     0.472   0.337  +0.135
    12     0.541   0.396  +0.145   <- peak manifold advantage (~+37%)
    16     0.605   0.494  +0.111
    24     0.686   0.682  +0.004   <- crossover
    32     0.730   0.790  -0.061
    64     0.737   0.969  -0.232   <- manifold SATURATES ~0.74; flat -> ~1.0

READING (CORRECTED — the dense end was a BUG/artifact, diagnosed):
  * FAIR regime k<=32: the softmax-TopK gate honors target_k (measured actual
    active/row: k=4->4, 8->8, 16->16, 24->24, 32->31.9). Here the result is real:
    the MANIFOLD wins in the SPARSE regime k=2..16 (gap peaks +0.145 ~+37% at
    k=12), they CROSS OVER near k=24, and the FLAT SAE wins at k=32 (0.790 vs
    0.730). Curvature is a sparse, per-active-atom efficiency; past ~k=24 the flat
    SAE's unconstrained linear atoms pull ahead even at honored k.
  * INVALID regime k>=48: the softmax-TopK gate CAPS the manifold's effective
    active atoms at ~36 regardless of target_k (measured: target_k=48 and 64 both
    give ~36.2 active/row), so the manifold never actually reaches k=48/64 active
    while the flat SAE does. The apparent "manifold saturates at ~0.74 while flat
    -> 0.97" is therefore NOT a manifold capacity limit (64 order-3 curves easily
    span R^64) — it is the gate cap PLUS dense overfitting (OOS_VE at k=64 DROPS
    with more steps: 0.735 @700 -> 0.707 @2000 -> 0.671 @5000). The k>=48 rows
    below are kept only to show the artifact; do not read them as a fair comparison.
  Bug to fix (flagged): the softmax_topk sparsity layer cannot activate more than
  ~36/64 atoms — for a fair dense comparison the manifold needs a gate that honors
  target_k (or an unbounded jumprelu-style gate). Does not affect the SPARSE
  regime (k<=32), which is how SAEs are actually used.

Run: saevenv/bin/python tests/sae/manifold_vs_flat_ksweep.py

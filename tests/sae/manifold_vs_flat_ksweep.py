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
"""

import os, sys, glob, time, numpy as np, torch
sys.path.insert(0,'/Users/user/gam')
import gamfit.torch as gt
from dictionary_learning.trainers.top_k import TopKTrainer

def load(name):
    z=np.load(f"/Users/user/gam/tests/data/{name}").astype(np.float64); n=z.shape[0]; ntr=(n*6)//10
    return torch.tensor(z[:ntr]), torch.tensor(z[ntr:])
def ve(xh,x,mtr): return 1-((xh-x)**2).sum().item()/((x-mtr)**2).sum().item()
def manifold(ztr,zte,D,k,steps=700,bs=128,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=gt.ManifoldSAEConfig(input_dim=ztr.shape[1],n_atoms=D,intrinsic_rank=1,
        atom_manifold="circle",atom_basis="fourier",basis_order=3,
        sparsity={"kind":"softmax_topk","target_k":k})
    sae=gt.ManifoldSAE(cfg).double(); opt=torch.optim.Adam(sae.parameters(),lr=2e-3)
    n=ztr.shape[0]; mtr=ztr.mean(0)
    for s in range(steps):
        x=ztr[torch.randint(0,n,(bs,))]; loss=((sae(x).x_hat-x)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sae.eval()
    with torch.no_grad(): return ve(sae(zte).x_hat,zte,mtr)
def flat(ztr,zte,D,k,steps=2500,bs=128,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    p=ztr.shape[1]; tr=TopKTrainer(steps=steps,activation_dim=p,dict_size=D,k=k,
        layer=0,lm_name="x",lr=None,device="cpu",seed=seed)
    n=ztr.shape[0]; ztrf=ztr.float(); ztef=zte.float()
    for s in range(steps): tr.update(s, ztrf[torch.randint(0,n,(bs,))])
    ae=tr.ae; ae.eval()
    with torch.no_grad(): return ve(ae(ztef).double(), zte, zte.mean(0))

ztr,zte=load("olmo_l18_pca64_635.npy")
D=64
print(f"== k-sweep, single layer OLMo l18, dict={D}, OOS var-expl (2 seeds avg) ==")
print(f"{'k':>4} {'MANIFOLD':>10} {'FLAT':>8} {'gap':>7}")
for k in [1,2,3,4,6,8,12,16,24,32,48,64]:
    mv=np.mean([manifold(ztr,zte,D,k,seed=s) for s in range(2)])
    fv=np.mean([flat(ztr,zte,D,k,seed=s) for s in range(2)])
    print(f"{k:>4} {mv:>10.3f} {fv:>8.3f} {mv-fv:>+7.3f}")
print("DONE")

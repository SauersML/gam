#!/usr/bin/env python3
"""Broad manifold-SAE vs flat-TopK-SAE exploration (both real trained SAEs).

FLAT  = dictionary_learning TopK SAE.  MANIFOLD = gamfit.torch.ManifoldSAE
(learned encoder, softmax-TopK, curved Fourier atoms, Rust-backed basis).
OOS variance explained on a 60/40 contiguous split.

FINDINGS (run on this machine; OLMo l18 train 381 / test 254):
  [1] On l18 the manifold beats flat by ~+44%, ROBUST across 3 seeds (low std):
        D=64 k=2 : MAN 0.320+/-.004  FLAT 0.222+/-.006
        D=64 k=4 : MAN 0.373+/-.010  FLAT 0.260+/-.010
        D=128 k=4: MAN 0.385+/-.004  FLAT 0.266+/-.011
  [2] PARAM-EFFICIENT, not just "more params": to catch manifold(64,k4)=0.373 a
      FLAT dict needs ~1024 atoms (0.370); at matched params (~512) flat is 0.326.
  [3] CAVEAT/BUG: setting basis_order alone did NOT change the basis (order 1..4
      all gave 0.368) — n_basis_per_atom (not set here) fixes the real basis size.
      So that sub-sweep is a no-op; treat order-independence as UNVERIFIED.
  [4] DATASET-DEPENDENT — NOT universal. On olmo_mixedlayer the FLAT SAE WINS:
        D=64 k=4 : MAN 0.629  FLAT 0.704
        D=128 k=8: MAN 0.633  FLAT 0.758
      The manifold helps where the data is curved (l18: autocorrelated, low local
      intrinsic dim) and hurts where it reconstructs well linearly (mixedlayer,
      higher EV ceiling). Confounder: the manifold may be under-trained on the
      larger mixedlayer set (768 rows, 600 steps vs flat 2500) — worth chasing.

Run:  saevenv/bin/python tests/sae/manifold_vs_flat_exploration.py
      (needs dictionary_learning + the repo on sys.path for gamfit/_rust.abi3.so)
"""

import os, sys, glob, time, numpy as np, torch
sys.path.insert(0,'/Users/user/gam')
import gamfit.torch as gt
from dictionary_learning.trainers.top_k import TopKTrainer

def load(name):
    z = np.load(f"/Users/user/gam/tests/data/{name}").astype(np.float64)
    n=z.shape[0]; ntr=(n*6)//10
    return torch.tensor(z[:ntr]), torch.tensor(z[ntr:])

def ve(xh,x,mtr): return 1 - ((xh-x)**2).sum().item()/((x-mtr)**2).sum().item()

def manifold(ztr,zte,D,k,rank=1,order=3,manif="circle",steps=600,bs=128,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=gt.ManifoldSAEConfig(input_dim=ztr.shape[1],n_atoms=D,intrinsic_rank=rank,
        atom_manifold=manif,atom_basis="fourier",basis_order=order,
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
    n=ztr.shape[0]; mtr=ztr.mean(0).float() if ztr.dtype==torch.float32 else ztr.mean(0)
    ztrf=ztr.float(); ztef=zte.float()
    for s in range(steps): tr.update(s, ztrf[torch.randint(0,n,(bs,))])
    ae=tr.ae; ae.eval()
    with torch.no_grad(): return ve(ae(ztef).double(), zte, zte.mean(0))

ztr,zte=load("olmo_l18_pca64_635.npy")
print(f"== OLMo l18  train {ztr.shape[0]} test {zte.shape[0]} ==\n")

print("[1] SEED ROBUSTNESS (3 seeds, mean+/-std OOS var-expl), matched (D,k):")
for (D,k) in [(64,2),(64,4),(128,4)]:
    mv=[manifold(ztr,zte,D,k,seed=s) for s in range(3)]
    fv=[flat(ztr,zte,D,k,seed=s) for s in range(3)]
    print(f"  D={D} k={k}:  MANIFOLD {np.mean(mv):.3f}+/-{np.std(mv):.3f}   FLAT {np.mean(fv):.3f}+/-{np.std(fv):.3f}")

print("\n[2] PARAM-EFFICIENCY: manifold(64,4,order3,circle) uses ~64*7=448 atom-equiv;")
print("    how big a FLAT dict (k=4) to catch its OOS?")
m644=np.mean([manifold(ztr,zte,64,4,seed=s) for s in range(3)])
print(f"    manifold(64,k4) OOS = {m644:.3f}")
for Dflat in [64,128,256,512,1024,2048]:
    fv=flat(ztr,zte,Dflat,4,seed=0)
    print(f"    flat dict={Dflat:>4} k=4 OOS = {fv:.3f}")

print("\n[3] MANIFOLD rank/order/manifold-type sweep at (D=64,k=4):")
for (rank,order,manif) in [(1,1,"circle"),(1,2,"circle"),(1,3,"circle"),(1,4,"circle"),
                            (2,2,"torus"),(2,3,"torus"),(2,2,"sphere")]:
    try:
        v=np.mean([manifold(ztr,zte,64,4,rank=rank,order=order,manif=manif,seed=s) for s in range(2)])
        print(f"    rank={rank} order={order} {manif:>7}: OOS {v:.3f}")
    except Exception as e:
        print(f"    rank={rank} order={order} {manif:>7}: ERR {str(e)[:60]}")

print("\n[4] SECOND DATASET olmo_mixedlayer (768x64):")
ztr2,zte2=load("olmo_mixedlayer_pca64_768.npy")
for (D,k) in [(64,4),(128,8)]:
    mv=np.mean([manifold(ztr2,zte2,D,k,seed=s) for s in range(2)])
    fv=np.mean([flat(ztr2,zte2,D,k,seed=s) for s in range(2)])
    print(f"  D={D} k={k}:  MANIFOLD {mv:.3f}   FLAT {fv:.3f}")
print("\nDONE")

"""
THE REAL manifold SAE (gamfit.sae_fit) on REAL Qwen3-8B L18, sink-peeled.
First goal: does the actual fitter RUN + converge on real data, and what atom topologies
does its structure search DISCOVER unsupervised? Report reconstruction EV + atom types.
Small (6k rows) first for a fast signal.
"""
import json, os, time, numpy as np

def required_env(name, message):
    try:
        return os.environ[name]
    except KeyError as exc:
        raise SystemExit(message) from exc

R=required_env("GAM_MSI_DATA", "set GAM_MSI_DATA to the activation-harvest root")
OUT=f"{R}/gam_ceiling_fable/experiments/real_manifold_sae"; os.makedirs(OUT, exist_ok=True)
import gamfit

# ---- real L18, sink-peeled ----
X=np.load(f"{R}/harvest_out/qwen3_8b_wikitext/resid_L18.npy", mmap_mode="r")
rng=np.random.default_rng(0); N=6000
idx=np.sort(rng.choice(X.shape[0], N, replace=False))
X=np.asarray(X[idx], dtype=np.float32)
mu=X.mean(0); Xc=X-mu
U,S,Vt=np.linalg.svd(Xc, full_matrices=False)
Xp=(Xc-(Xc@Vt[:1].T)@Vt[:1]).astype(np.float32)   # peel top sink PC
print(f"data: {Xp.shape}, sink var frac {(S[0]**2/(S**2).sum()):.3f}", flush=True)
tot=float((Xp**2).sum())
def ev(recon): return 1.0-float(((Xp-recon)**2).sum())/tot

t0=time.time()
print("running gamfit.sae_fit (structure search ON, discovers topologies)...", flush=True)
res=gamfit.sae_fit(Xp, config=dict(K=16, d_atom=2, n_iter=12))
dt=time.time()-t0
print(f"FIT DONE in {dt:.1f}s", flush=True)

print("=== summary ===", flush=True)
summ=res.get("summary", {})
print(json.dumps({k:(v if isinstance(v,(int,float,str,bool)) else str(type(v).__name__)) for k,v in summ.items()}, indent=1)[:1500], flush=True)

# atom topologies discovered
atoms=res.get("atoms", [])
topo_counts={}
for a in atoms:
    t=getattr(a,"topology",None) or getattr(a,"manifold",None) or getattr(a,"kind",None) or type(a).__name__
    t=str(t); topo_counts[t]=topo_counts.get(t,0)+1
print("=== discovered atom topologies ===", flush=True)
print(topo_counts, flush=True)

# reconstruction EV if available
model=res.get("model", None)
recon_ev=None
for attr in ("reconstruction","reconstruct","predict"):
    if hasattr(model, attr):
        try:
            rec=getattr(model,attr)(Xp); recon_ev=ev(np.asarray(rec)); break
        except Exception as e:
            print(f"  {attr} failed: {e}", flush=True)
if recon_ev is None and "reconstruction" in res:
    try: recon_ev=ev(np.asarray(res["reconstruction"]))
    except Exception: pass
print(f"=== manifold-SAE reconstruction EV: {recon_ev} ===", flush=True)

# model attributes (to learn the API for the head-to-head)
print("model type:", type(model).__name__, flush=True)
print("model attrs:", [a for a in dir(model) if not a.startswith("_")][:40], flush=True)
print("res keys:", list(res.keys()), flush=True)

out=dict(N=N, fit_seconds=dt, topo_counts=topo_counts, recon_ev=recon_ev,
         summary={k:str(v)[:120] for k,v in summ.items()})
json.dump(out, open(f"{OUT}/real_sae_probe.json","w"), indent=1)
print("saved probe json", flush=True)

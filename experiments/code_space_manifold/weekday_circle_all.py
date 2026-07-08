"""Supervised weekday-circle falsifier + plottable geometry dump. Loads the DOSE
weekday battery activations (70 = 10 templates x 7 weekdays) at L11/L17/L23 of
Qwen3.6-35B and tests whether the 7 weekday centroids trace a RING at two levels:
RAW activation space (does the MODEL encode it) and SAE CODE space (does the K=32000
dict preserve it). Ring metric = Pearson corr(calendar ring-distance, centroid
distance) over the 21 day-pairs + a 20k-permutation-null p-value + angular
calendar-order recovery. Dumps ONE npz with per-layer per-token 2D projections (top-2
centroid-PCA plane) so the ACTUAL circle can be plotted, and a summary JSON. Also
reports the harvest<->dict recon-EV so the code-space test's validity is checkable."""
import sys, json
import numpy as np
ROOT = "/projects/standard/hsiehph/sauer354"
sys.path.insert(0, ROOT + "/msae_l17/driver")
from compose_l17_stagewise import _load_tier0, _apply_tier0

WD = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
# per-layer dict + tier0 (code panel only when the dict exists)
DICTS = {
 11: (ROOT+"/msae_l17/t1_out/decoder_L11_K32000.npy", ROOT+"/msae_l17/tier0_L11.json"),
 17: (ROOT+"/msae_l17/t1_out/decoder_K32000.npy",     ROOT+"/msae_l17/tier0_recentered.json"),
 23: (ROOT+"/msae_l17/t1_out/decoder_L23_K32000.npy", ROOT+"/msae_l17/tier0_L23.json"),
}

def ring_dist():
    d = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            d[i,j] = min(abs(i-j), 7-abs(i-j))
    return d
RD = ring_dist()[np.triu_indices(7,1)]

def ring_stats(cent, n_perm=20000, seed=0):
    iu = np.triu_indices(7,1)
    def corr(order):
        C = cent[order]
        dd = np.linalg.norm(C[:,None,:]-C[None,:,:],axis=2)[iu]
        return 0.0 if dd.std()<1e-12 else float(np.corrcoef(RD,dd)[0,1])
    obs = corr(np.arange(7)); rng=np.random.default_rng(seed); ge=1
    for _ in range(n_perm):
        if corr(rng.permutation(7))>=obs: ge+=1
    return obs, ge/(n_perm+1)

def plane_and_proj(cent, tokens):
    """top-2 PCA plane of the 7 centroids; project centroids + all tokens onto it."""
    mu = cent.mean(0); C = cent-mu
    U,S,Vt = np.linalg.svd(C, full_matrices=False)
    var2 = float((S[:2]**2).sum()/(S**2).sum())
    P = Vt[:2]                                   # (2,d)
    cproj = (cent-mu)@P.T; tproj = (tokens-mu)@P.T
    ang = np.arctan2(cproj[:,1], cproj[:,0]); order=list(np.argsort(ang))
    def is_cal(seq):
        seq=list(seq)
        for s in (seq, seq[::-1]):
            db=s+s
            for r in range(7):
                if db[r:r+7]==list(range(7)): return True
        return False
    return cproj, tproj, var2, [WD[i] for i in order], bool(is_cal(order))

def centroids(V, labels):
    return np.stack([V[labels==w].mean(0) for w in range(7)])

def t1_codes(D, X, active=32):
    """T1-exact: top-|score| selection + active-set ridge-LS amplitudes (NOT raw
    projection — raw scores double-count correlated atoms and blow up the recon)."""
    s = X @ D.T
    idx = np.argpartition(-np.abs(s), active-1, axis=1)[:, :active]
    codes = np.zeros((X.shape[0], D.shape[0]), np.float32)
    Dg = D[idx]                                          # (m,a,P)
    G = np.einsum("map,mbp->mab", Dg, Dg)
    ridge = 1e-6 * np.trace(G, axis1=1, axis2=2)[:, None, None] * np.eye(active)[None]
    rhs = np.einsum("map,mp->ma", Dg, X.astype(np.float32))
    c = np.linalg.solve(G + ridge, rhs[..., None])[..., 0]   # (m,a)
    rows = np.arange(X.shape[0])[:, None]
    codes[rows, idx] = c.astype(np.float32)
    return codes, idx

def main():
    z = np.load(ROOT+"/weekday_acts_q36.npz")
    labels = np.asarray(z["labels"])
    dump = {"labels": labels, "weekdays": np.array(WD)}
    summary = {}
    import os
    for L in (11,17,23):
        key=f"acts_L{L}"
        if key not in z.files: continue
        acts = np.asarray(z[key], np.float64)
        print(f"\n=== L{L}  acts {acts.shape} ===", flush=True)
        # RAW
        cent = centroids(acts, labels)
        corr,p = ring_stats(cent)
        cproj,tproj,var2,order,iscal = plane_and_proj(cent, acts)
        print(f" RAW : ring_corr={corr:+.3f} perm_p={p:.4f} top2Dvar={var2:.3f} "
              f"order={'-'.join(order)} calendar_cyclic={iscal}", flush=True)
        dump[f"L{L}_raw_token2d"]=tproj; dump[f"L{L}_raw_centroid2d"]=cproj
        dump[f"L{L}_raw_token_angle"]=np.arctan2(tproj[:,1],tproj[:,0])
        srec={"raw":{"ring_corr":corr,"perm_p":p,"top2D_var":var2,
                     "angular_order":order,"calendar_cyclic":iscal}}
        # CODE (if dict exists)
        dpath,t0path = DICTS[L]
        if os.path.exists(dpath) and os.path.exists(t0path):
            D = np.ascontiguousarray(np.load(dpath), np.float32)
            mean,scale=_load_tier0(t0path)
            Xt=_apply_tier0(acts.astype(np.float32), mean, scale)
            codes,idx=t1_codes(D,Xt,32)
            # recon-EV (space-match validity): baseline=zero on tier0 space
            recon = codes@D  # (70,d)
            ev = 1.0 - float(((Xt-recon)**2).sum())/float((Xt**2).sum())
            ccent=centroids(codes,labels)
            ccorr,cp=ring_stats(ccent)
            ccproj,ctproj,cvar2,corder,ciscal=plane_and_proj(ccent,codes)
            print(f" CODE: ring_corr={ccorr:+.3f} perm_p={cp:.4f} top2Dvar={cvar2:.3f} "
                  f"order={'-'.join(corder)} calendar_cyclic={ciscal}  recon_EV={ev:.3f}", flush=True)
            dump[f"L{L}_code_token2d"]=ctproj; dump[f"L{L}_code_centroid2d"]=ccproj
            srec["code"]={"ring_corr":ccorr,"perm_p":cp,"top2D_var":cvar2,
                          "angular_order":corder,"calendar_cyclic":ciscal,"recon_ev":ev}
        else:
            print(f" CODE: dict not ready ({dpath})", flush=True)
        summary[f"L{L}"]=srec
    out_npz=ROOT+"/gam/experiments/code_space_manifold/weekday_probe_raw.npz"
    import os; os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **dump)
    json.dump(summary, open(ROOT+"/weekday_circle_summary.json","w"), indent=2)
    print("\nwrote", out_npz)
    print("SUMMARY", json.dumps(summary))

if __name__=="__main__":
    main()

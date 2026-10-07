"""Budget descent on vpd4l's MLPs with intrinsic expected gates (#2951 prototype for the Rust fit).

P replaces each of the 8 MLP maps (c_fc, down_proj of 4 layers) by slices: slice i of a map has
read v_i (a column of V) and write u_i (a row of U), and contributes (v_i.x) u_i on the map's own
input x, times its gate. Attention is M's, exact. No gating network: slice i's gate depends only on
its own read, r_i = |v_i.x| ||u_i|| (the norm of the slice's output on x), against its own
threshold tau_i:
  training    the expected gate Phi((r_i - tau_i) / s_i) under Gaussian threshold noise of fixed
              scale s_i (0.1 x the root mean square of r_i over the fit tokens at the start)
  evaluation  the hard gate 1[r_i > tau_i]
Objective: KL(M || P) in bits per token on training tokens + lambda (E[k] - K), where
E[k] = sum_i Phi over all 8 maps (expected active slices per token, rank-one equivalents);
lambda >= 0 by dual ascent with library_mdl's measured step (below) until E[k] <= K.
Starts:
  svd  the input-whitened SVD of each map (W C^(1/2) = U S V^T on M's inputs of rows 0..15;
       reads C^(-1/2) v, writes s u; all on = M), thresholds 3 s_i below zero, so every gate is on
  vpd  VPD's MLP slices (s-55ea3f9b), one threshold per map set so the active count on the fit
       tokens matches VPD's causal-importance count at that map (fitmath's KL 3.28 start), then
       trained per slice
  vpdgroup  as vpd, with each down_proj slice tied to the gate of the c_fc slice of its layer whose
       firing is nearest its own (below)
Gate arms (DESCENT_ARM): own (above); dir, a separate signed gate direction per slice; router, a
small causal router per layer added to the own read; router_pure, the router alone (below).
Dual step (DESCENT_DUAL): measured (default, library_mdl's rule, below) or fixed (lambda <- lambda +
0.01 log(E[k] / K) on the loss KL + lambda log E[k], the first runs' rule).
Training rows 0..1023 of tokens.f64, held-out evaluation rows 1024..1031 (4096 tokens), where VPD's
causal-importance masks give KL 0.737 at 129 active MLP slices per token.
Usage: budget_descent.py START K STEPS OUT.json GATE EVAL SECONDS GAM TARGET VPD TOKENS
  GATE     mf = expected gate forward and derivative; st = hard gate forward, expected gate derivative
  EVAL     steps between held-out evaluations; SECONDS: wall-clock limit of the training loop
  GAM      bench/vpd_2951 (vpd_model), TARGET the target run (t-9d2b8f02), VPD the decomposition
           export's export.json (engine/vpd4l_decomposition), TOKENS engine/vpd4l_pile2p27/tokens.f64"""
import sys, json, math, time
import numpy as np, scipy.linalg as sl, torch, torch.nn.functional as F

start, K, steps, out, gate = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
EVAL, LIMIT = int(sys.argv[6]), float(sys.argv[7])
sys.path.insert(0, sys.argv[8])
import vpd_model
from vpd_model import load_target, site_names
from pathlib import Path
vpd_model.TARGET_DIR = Path(sys.argv[9])
if not (vpd_model.TARGET_DIR / 'model_step_99999.safetensors').exists():  # the MATS copy holds the .pt
    vpd_model.load_file = lambda f: torch.load(f.replace('.safetensors', '.pt'), map_location='cpu', weights_only=True)
VPD_DIR, TOKENS = (Path(sys.argv[10]) if sys.argv[10].endswith('.pth') else Path(sys.argv[10]).parent), sys.argv[11]
import os
# DESCENT_ARM=dir: each slice's gate reads its own direction g_i (signed), z = (g_i.x - tau_i)/s_i,
# with g_i started at the slice's read v_i ||u_i||, signed so its firing on M's fit tokens is kept.
ARM = os.environ.get('DESCENT_ARM', 'own')
dev = os.environ.get('DESCENT_DEV') or ('cuda' if torch.cuda.is_available() else 'mps')
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
T = load_target(dev)
mlp = [n for n in site_names() if '.mlp.' in n]
tok = np.memmap(TOKENS, dtype=np.uint16 if TOKENS.endswith('.u16') else np.float64, mode='r').reshape(-1, 513)
ev = torch.tensor(tok[1024:1032, :512].astype(np.int64), device=dev)
# Training rows: DESCENT_TRAIN_ROWS rows of the file, skipping the held-out rows 1024..1031 (default 1024).
train_rows, batch, seq = int(os.environ.get('DESCENT_TRAIN_ROWS', '1024')), 8, 256
# VPD's mean active (causal importance > 0) slices per token at each MLP map, rows 1024..1031,
# from M's clean inputs (fitmath proto.log); 129 in total.
VPD_COUNTS = {'h.0.mlp.c_fc': 18.62, 'h.0.mlp.down_proj': 20.20, 'h.1.mlp.c_fc': 4.23,
              'h.1.mlp.down_proj': 3.20, 'h.2.mlp.c_fc': 7.16, 'h.2.mlp.down_proj': 7.77,
              'h.3.mlp.c_fc': 26.43, 'h.3.mlp.down_proj': 41.07}

@torch.no_grad()
def site_inputs(ids):
    xs = {n: [] for n in mlp}
    for i in range(0, ids.shape[0], 4):
        for n in mlp: T.site(n).cache_input = True
        T(ids[i:i + 4])
        for n in mlp:
            st = T.site(n); xs[n].append(st.last_input.reshape(-1, st.last_input.shape[-1])); st.cache_input = False
    return {n: torch.cat(v) for n, v in xs.items()}

X = site_inputs(torch.tensor(tok[0:16, :512].astype(np.int64), device=dev))
vpdlike = start in ('vpd', 'vpdgroup')
if vpdlike and str(VPD_DIR).endswith('.pth'):
    raw = torch.load(str(VPD_DIR), map_location='cpu', weights_only=True, mmap=True)
    load = lambda k: raw['_components.' + k.rsplit('.', 1)[0].replace('.', '-') + '.' + k.rsplit('.', 1)[1]].float().to(dev)
elif vpdlike:
    shapes = {k: v['shape'] for k, v in json.load(open(VPD_DIR / 'export.json'))['files'].items()}
    load = lambda k: torch.tensor(np.fromfile(VPD_DIR / f'{k}.f64', dtype='<f8').reshape(shapes[k]), dtype=torch.float32, device=dev)
P = {}
for n in mlp:
    W = T.site(n).W
    if start == 'svd':
        x = X[n].cpu().double().numpy(); Wd = W.cpu().double().numpy()
        C = x.T @ x / x.shape[0]; C += 1e-6 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        lam_, Q = sl.eigh(C)
        Us, S, Vt = sl.svd(Wd @ ((Q * np.sqrt(lam_)) @ Q.T), full_matrices=False)
        V = torch.tensor(((Q / np.sqrt(lam_)) @ Q.T) @ Vt.T, dtype=torch.float32, device=dev)
        U = torch.tensor((Us * S).T, dtype=torch.float32, device=dev)
    else:
        V, U = load(n + '.V'), load(n + '.U')
    with torch.no_grad():
        r = (X[n] @ V).abs() * U.norm(dim=1)
        s = 0.1 * r.pow(2).mean(0).sqrt().clamp_min(1e-12)
        if start == 'svd':
            tau = -3 * s
        else:
            q = 1 - VPD_COUNTS[n] / V.shape[1]
            flat = r.reshape(-1)
            idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            tau = torch.full_like(s, torch.quantile(flat[idx], q).item())
    P[n] = {'V': V.clone().requires_grad_(), 'U': U.clone().requires_grad_(), 'tau': tau.clone().requires_grad_(), 's': s}
    if ARM == 'dir':
        with torch.no_grad():
            c = X[n] @ V
            sgn = torch.sign((c * (r > tau).float()).sum(0)); sgn[sgn == 0] = 1
            P[n]['G'] = (V * (sgn * U.norm(dim=1))[None, :]).clone().requires_grad_()
    print(n, 'slices', V.shape[1], 'sum error', float((V @ U - W.T).abs().max()), flush=True)
# vpdgroup: a down_proj slice whose firing on M's fit tokens is closer (Hamming distance) to one of
# its layer's c_fc slices than to never firing is tied to that c_fc slice: one gate, the c_fc
# slice's own, for the group. A group on counts 1 + its tied down slices (rank-one equivalents).
GROUP, MULT = {}, {}
if start == 'vpdgroup':
    with torch.no_grad():
        for l in range(4):
            fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
            on = lambda m: (((X[m] @ P[m]['V']).abs() * P[m]['U'].norm(dim=1)) > P[m]['tau']).float()
            gf, gd = on(fc), on(dn)
            mism = gf.sum(0)[:, None] + gd.sum(0)[None, :] - 2 * (gf.T @ gd)
            bm, best = mism.min(0)
            owner = torch.where(bm < gd.sum(0), best, torch.full_like(best, -1))
            GROUP[dn] = (fc, owner)
            MULT[fc] = 1 + torch.bincount(owner[owner >= 0], minlength=gf.shape[1]).float()
            print(dn, 'tied', int((owner >= 0).sum()), 'of', owner.numel(), 'to', int((MULT[fc] > 1).sum()), 'c_fc slices', flush=True)
# DESCENT_ARM=router: each MLP layer gets a small causal router on the layer's own input x (the c_fc
# input at that position only): z_i = (r_i + [G2 relu(G1 x)]_i - tau_i) / s_i for every slice of the
# layer's two maps, G1 of shape d x 64 shared by the layer, G2 of shape 64 x C per map, started at
# zero (so the start is the own-read gate's).
# DESCENT_ARM=router_pure: the router alone, z_i = ([G2 relu(G1 x)]_i - tau_i) / s_i, with G2 started
# at the least-squares fit of the own reads r on M's fit tokens from the 64 router features.
ROUTER = {}
if ARM in ('router', 'router_pure'):
    for l in range(4):
        fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
        d = P[fc]['V'].shape[0]
        G1 = torch.randn(d, 64, device=dev) / math.sqrt(d)
        R = {'G1': G1.requires_grad_()}
        for n in (fc, dn):
            if ARM == 'router':
                R[n] = torch.zeros(64, P[n]['V'].shape[1], device=dev, requires_grad=True)
            else:
                with torch.no_grad():
                    H = torch.relu(X[fc] @ G1)
                    r = (X[n] @ P[n]['V']).abs() * P[n]['U'].norm(dim=1)
                    R[n] = torch.linalg.lstsq(H.cpu().double(), r.cpu().double()).solution.float().to(dev).requires_grad_()
        ROUTER[l] = R
del X

state = {'mode': 'M', 'soft': [], 'hard': [], 'gates': {}, 'route': {}}
SQ2 = math.sqrt(2)
def make(n):
    st = T.site(n); p = P[n]; grp = GROUP.get(n); mult = MULT.get(n)
    layer = int(n.split('.')[1]); router = ROUTER.get(layer)
    if grp is not None:
        tied, own = grp[1] >= 0, grp[1].clamp_min(0)
    def fwd(x):
        if state['mode'] == 'M':
            return x @ st.W.T
        c = x @ p['V']
        if state['mode'] == 'all':
            return c @ p['U']
        read = x @ p['G'] if ARM == 'dir' else c.abs() * p['U'].norm(dim=1)
        if router is not None:
            if n.endswith('c_fc'):
                state['route'][layer] = torch.relu(x @ router['G1'])
            read = state['route'][layer] @ router[n] if ARM == 'router_pure' else read + state['route'][layer] @ router[n]
        z = (read - p['tau']) / p['s']
        hard = (z > 0).float()
        phi = 0.5 * (1 + torch.erf(z / SQ2))
        w = 1.0
        if grp is not None:
            fh, fp = state['gates'][grp[0]]
            hard = torch.where(tied, fh[..., own], hard); phi = torch.where(tied, fp[..., own], phi)
            w = (~tied).float()
        if mult is not None:
            state['gates'][n] = (hard, phi); w = mult
        state['hard'].append((hard * w).sum(-1).reshape(-1))
        if state['mode'] == 'hard':
            return (c * hard) @ p['U']
        state['soft'].append((phi * w).sum(-1).reshape(-1))
        g = phi if gate == 'mf' else hard + phi - phi.detach()
        return (c * g) @ p['U']
    return fwd
for n in mlp: T.site(n)._forward = make(n)

def kl_bits(lm, lp):
    pm = F.log_softmax(lm.float(), -1); pp = F.log_softmax(lp.float(), -1)
    return (pm.exp() * (pm - pp)).sum(-1) / math.log(2)

def run(ids, mode):
    state['mode'], state['soft'], state['hard'] = mode, [], []
    return T(ids)

@torch.no_grad()
def evaluate():
    r = {'kl': [], 'kl_soft': [], 'kl_all_on': [], 'active': [], 'active_soft': [], 'per_map': []}
    for i in range(0, ev.shape[0], 4):
        ids = ev[i:i + 4]
        lm = run(ids, 'M')
        lp = run(ids, 'hard'); r['kl'].append(kl_bits(lm, lp).mean().item())
        r['active'].append(torch.stack(state['hard']).sum(0).mean().item())
        r['per_map'].append([h.mean().item() for h in state['hard']])
        lp = run(ids, 'soft'); r['kl_soft'].append(kl_bits(lm, lp).mean().item())
        r['active_soft'].append(torch.stack(state['soft']).sum(0).mean().item())
        lp = run(ids, 'all'); r['kl_all_on'].append(kl_bits(lm, lp).mean().item())
    out = {k: float(np.mean(v)) for k, v in r.items() if k != 'per_map'}
    out['per_map'] = [round(float(x), 2) for x in np.mean(r['per_map'], 0)]
    return out

params = [P[n][w] for n in mlp for w in (('V', 'U', 'G') if ARM == 'dir' else ('V', 'U'))]
# Adam steps of 0.3% of each tensor's root mean square, thresholds 1% of their site's noise scale x 10.
# DESCENT_LR multiplies every step size (default 1).
LR = float(os.environ.get('DESCENT_LR', '1'))
groups = [{'params': [q], 'lr': LR * 3e-3 * q.detach().pow(2).mean().sqrt().item()} for q in params]
groups += [{'params': [P[n]['tau']], 'lr': LR * 0.1 * P[n]['s'].mean().item()} for n in mlp]
# Router: G1 at 0.3% of its root mean square, G2 at 0.3% of its map's noise scale.
for l, R in ROUTER.items():
    groups.append({'params': [R['G1']], 'lr': LR * 3e-3 * R['G1'].detach().pow(2).mean().sqrt().item()})
    groups += [{'params': [R[n]], 'lr': LR * 3e-3 * P[n]['s'].mean().item()} for n in R if n != 'G1']
opt = torch.optim.Adam(groups)
trainable = [q for g in groups for q in g['params']]
# The budget's multiplier (library_mdl's rule, Settings::budget): the step descends KL + lam (E[k] - K),
# then lam <- max(0, lam + eta (E[k] - K)) with eta = lam_hat / (K B), lam_hat = |<g_F, g_k>| / |g_k|^2
# the multiplier at which the budget's gradient cancels the KL gradient's component along
# g_k = dE[k]/dtheta (both measured on this step), B the batches in one pass over the training rows.
B = train_rows * 512 / (batch * seq)
DUAL = os.environ.get('DESCENT_DUAL', 'measured')
lam, rng = 0.0, np.random.default_rng(0)
log = {'start': start, 'K': K, 'steps': steps, 'gate': gate, 'arm': ARM, 'dual': DUAL, 'train_rows': train_rows, 'trace': []}
e = evaluate(); print('start', e, flush=True); log['trace'].append({'step': 0, **e})
t0 = time.time()
for step in range(steps):
    rows = rng.integers(0, train_rows, batch); rows = np.where(rows >= 1024, rows + 8, rows); offs = rng.integers(0, 513 - seq, batch)
    ids = torch.tensor(np.stack([tok[r, o:o + seq] for r, o in zip(rows, offs)]).astype(np.int64), device=dev)
    with torch.no_grad():
        lm = run(ids, 'M')
    lp = run(ids, 'soft')
    kl = kl_bits(lm, lp).mean()
    ek = torch.stack(state['soft']).sum(0).mean()
    hk = torch.stack(state['hard']).sum(0).mean().item()
    if DUAL == 'fixed':
        opt.zero_grad(); (kl + lam * torch.log(ek)).backward(); opt.step()
        lam = max(0.0, lam + 0.01 * math.log(ek.item() / K))
        g_f = None
    else:
        g_f = torch.autograd.grad(kl, trainable, retain_graph=True, allow_unused=True)
    if g_f is not None:
        g_k = torch.autograd.grad(ek, trainable, allow_unused=True)
        along = sum((a * b).sum() for a, b in zip(g_f, g_k) if a is not None and b is not None).item()
        square = sum(b.pow(2).sum() for b in g_k if b is not None).item()
        for q, a, b in zip(trainable, g_f, g_k):
            q.grad = (torch.zeros_like(q) if a is None else a) + (0 if b is None else lam * b)
        opt.step()
        if square > 0:
            lam = max(0.0, lam + abs(along) / square / (K * B) * (ek.item() - K))
    last = step == steps - 1 or time.time() - t0 > LIMIT
    if (step + 1) % EVAL == 0 or last:
        e = evaluate()
        rec = {'step': step + 1, 'lambda': lam, 'train_kl': kl.item(), 'train_k_soft': ek.item(), 'train_k_hard': hk, **e,
               'seconds': time.time() - t0}
        log['trace'].append(rec); print(rec, flush=True)
        json.dump(log, open(out, 'w'), indent=1)
    if last:
        break

"""Grow from empty on vpd4l's MLPs (#2951, descent): the opposite direction to budget descent.

P starts with no MLP parts (every MLP map's output 0; attention exact) and repeatedly adds the one
candidate slice that most reduces KL(M || P) per unit of the per-token budget, with the slice's
intrinsic gate (on iff r = |v.x| ||u|| > tau, its own read on P's own input) fitted on the spot:
  1. On a fixed fit batch, run P and record each MLP map's input x and output y_P. The map's residual
     is R = W x - y_P (M's map applied to P's own input, minus P's output).
  2. Every not-yet-added slice i of the map would add g_t (v_i.x_t) u_i on token t. Its threshold is
     the one that most reduces sum_t |R_t - g_t (v_i.x_t) u_i|^2 over thresholds on its own read
     (tokens sorted by r_i; the best prefix); its rate p_i is that prefix's share of the tokens.
  3. The 24 candidates with the largest such reduction per unit of rate, as a share of the map's
     output energy sum_t |W x_t|^2, are each added alone, and the 4 best c_fc candidates each with the
     2 best down_proj slices of their layer given them (a c_fc slice changes nothing until a down_proj
     slice reads its neuron: parts of two maps); KL(M || P) and the active count on the fit batch are
     measured for each, and the one with the largest measured KL reduction per added active count
     per token is kept.
Growth stops when the expected active count per token on the fit batch reaches K or no candidate
lowers the measured KL. Held-out KL (rows 1024..1031, hard gates, as budget_descent.py) is evaluated
whenever the active count crosses 8, 16, 32, 64 and 129, and at the end.
Candidates: VPD's MLP slices (s-55ea3f9b), as in budget_descent.py's vpd start.
Usage: grow_from_empty.py K OUT.json SECONDS GAM TARGET VPD TOKENS
  VPD is the decomposition export's export.json or VPD's model_400000.pth."""
import sys, json, math, time, os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

K, out, LIMIT = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
sys.path.insert(0, sys.argv[4])
import vpd_model
from vpd_model import load_target, site_names
vpd_model.TARGET_DIR = Path(sys.argv[5])
if not (vpd_model.TARGET_DIR / 'model_step_99999.safetensors').exists():  # the MATS copy holds the .pt
    vpd_model.load_file = lambda f: torch.load(f.replace('.safetensors', '.pt'), map_location='cpu', weights_only=True)
VPD, TOKENS = sys.argv[6], sys.argv[7]
dev = os.environ.get('DESCENT_DEV') or ('cuda' if torch.cuda.is_available() else 'mps')
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
T = load_target(dev)
mlp = [n for n in site_names() if '.mlp.' in n]
tok = np.memmap(TOKENS, dtype=np.uint16 if TOKENS.endswith('.u16') else np.float64, mode='r').reshape(-1, 513)
ev = torch.tensor(tok[1024:1032, :512].astype(np.int64), device=dev)
fit = torch.tensor(tok[0:4, :512].astype(np.int64), device=dev)  # 2048 fit tokens
if VPD.endswith('.pth'):
    raw = torch.load(VPD, map_location='cpu', weights_only=True, mmap=True)
    load = lambda n, w: raw['_components.' + n.replace('.', '-') + '.' + w].float().to(dev)
else:
    shapes = {k: v['shape'] for k, v in json.load(open(VPD))['files'].items()}
    load = lambda n, w: torch.tensor(np.fromfile(Path(VPD).parent / f'{n}.{w}.f64', dtype='<f8').reshape(shapes[f'{n}.{w}']),
                                     dtype=torch.float32, device=dev)
P = {}
for n in mlp:
    V, U = load(n, 'V'), load(n, 'U')
    P[n] = {'V': V, 'U': U, 'un': U.norm(dim=1), 'added': torch.zeros(V.shape[1], dtype=torch.bool, device=dev),
            'tau': torch.full((V.shape[1],), float('inf'), device=dev)}

state = {'mode': 'M', 'cache': None, 'count': []}
def make(n):
    st = T.site(n); p = P[n]
    def fwd(x):
        if state['mode'] == 'M':
            return x @ st.W.T
        c = x @ p['V']
        g = (p['added'] & (c.abs() * p['un'] > p['tau'])).float()
        state['count'].append(g.sum(-1).reshape(-1))
        y = (c * g) @ p['U']
        if state['cache'] is not None:
            state['cache'][n] = (x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        return y
    return fwd
for n in mlp: T.site(n)._forward = make(n)

def kl_bits(lm, lp):
    pm = F.log_softmax(lm.float(), -1); pp = F.log_softmax(lp.float(), -1)
    return ((pm.exp() * (pm - pp)).sum(-1) / math.log(2)).mean().item()

@torch.no_grad()
def run(ids, mode, cache=False):
    state['mode'], state['count'], state['cache'] = mode, [], ({} if cache else None)
    return T(ids)

@torch.no_grad()
def evaluate():
    kls, act = [], []
    for i in range(0, ev.shape[0], 4):
        lm = run(ev[i:i + 4], 'M'); lp = run(ev[i:i + 4], 'P')
        kls.append(kl_bits(lm, lp)); act.append(torch.stack(state['count']).sum(0).mean().item())
    return {'kl': float(np.mean(kls)), 'active': float(np.mean(act)),
            'added': {n: int(P[n]['added'].sum()) for n in mlp}}

with torch.no_grad():
    lm_fit = run(fit, 'M')
    energy = {}
    run(fit, 'P', cache=True)
    for n in mlp:
        x, _ = state['cache'][n]
        energy[n] = (x @ T.site(n).W.T).pow(2).sum().item()

log = {'K': K, 'trace': [], 'rounds': []}
e = evaluate(); print('start', e, flush=True); log['trace'].append({'round': 0, **e})
marks, t0, rnd = [8, 16, 32, 64, 129], time.time(), 0
def measure():
    """KL(M || P) and the active count per token on the fit batch."""
    lp = run(fit, 'P')
    return kl_bits(lm_fit, lp), torch.stack(state['count']).sum(0).mean().item()

def local(n):
    """Each not-yet-added slice of map n: its best threshold on its own read and the reduction of the
    map's squared residual it buys per unit of rate, as a share of the map's output energy (from the
    cache of the last run)."""
    p = P[n]; x, y = state['cache'][n]
    R = x @ T.site(n).W.T - y
    c = x @ p['V']; r = c.abs() * p['un']
    gain = 2 * c * (R @ p['U'].T) - c.pow(2) * p['un'].pow(2)
    order = r.argsort(0, descending=True)
    best, idx = gain.gather(0, order).cumsum(0).max(0)
    rate = (idx + 1).float() / x.shape[0]
    tau = r.gather(0, order).gather(0, (idx + 1).clamp(max=x.shape[0] - 1)[None]).squeeze(0)
    tau = torch.where(idx + 1 >= x.shape[0], torch.full_like(tau, -1.0), tau)
    score = best / energy[n] / rate
    score[p['added'] | (best <= 0)] = -float('inf')
    return score, tau

def put(items, on):
    for n, i, tau in items:
        P[n]['added'][i] = on; P[n]['tau'][i] = tau if on else float('inf')

kl_now, act_now = measure()
while True:
    rnd += 1
    with torch.no_grad():
        run(fit, 'P', cache=True)
        scored = {n: local(n) for n in mlp}
        singles = []
        for n, (score, tau) in scored.items():
            top = score.topk(min(24, score.numel()))
            singles += [(v.item(), [(n, i.item(), tau[i].item())]) for v, i in zip(top.values, top.indices) if v > -float('inf')]
        singles = sorted(singles, key=lambda t: -t[0])[:24]
        # Parts across the two maps of a layer: a c_fc slice alone changes nothing until a down_proj
        # slice reads its neuron, so the 4 best c_fc candidates are each tried with the 2 best
        # down_proj slices of their layer given them.
        pairs = []
        fc_c = sorted([(v, it) for v, it in singles if it[0][0].endswith('c_fc')], key=lambda t: -t[0])[:4]
        if not fc_c:
            for n in mlp:
                if n.endswith('c_fc'):
                    score, tau = scored[n]
                    top = score.topk(4)
                    fc_c += [(v.item(), [(n, i.item(), tau[i].item())]) for v, i in zip(top.values, top.indices) if v > -float('inf')]
            fc_c = sorted(fc_c, key=lambda t: -t[0])[:4]
        for _, it in fc_c:
            put(it, True)
            run(fit, 'P', cache=True)
            dn = it[0][0].replace('c_fc', 'down_proj')
            score, tau = local(dn)
            top = score.topk(2)
            pairs += [it + [(dn, i.item(), tau[i].item())] for v, i in zip(top.values, top.indices) if v > -float('inf')]
            put(it, False)
        best_c = None
        for items in [it for _, it in singles] + pairs:
            put(items, True)
            kl, act = measure()
            put(items, False)
            if kl < kl_now and act > act_now:
                rate = (kl_now - kl) / (act - act_now)
                if best_c is None or rate > best_c[0]:
                    best_c = (rate, items, kl, act)
    if best_c is None:
        print('no candidate lowers the KL', flush=True)
        break
    _, items, kl_now, act_now = best_c
    put(items, True)
    log['rounds'].append({'round': rnd, 'added': [[n, i] for n, i, _ in items], 'fit_kl': kl_now, 'fit_active': act_now})
    if rnd % 25 == 0:
        print(log['rounds'][-1], round(time.time() - t0), flush=True)
    done = act_now >= K or time.time() - t0 > LIMIT
    while marks and act_now >= marks[0]:
        e = evaluate(); rec = {'round': rnd, 'mark': marks.pop(0), 'fit_kl': kl_now, 'fit_active': act_now, **e, 'seconds': time.time() - t0}
        log['trace'].append(rec); print(rec, flush=True); json.dump(log, open(out, 'w'), indent=1)
    if done:
        break
e = evaluate(); rec = {'round': rnd, 'final': True, 'fit_kl': kl_now, **e, 'seconds': time.time() - t0}
log['trace'].append(rec); print(rec, flush=True); json.dump(log, open(out, 'w'), indent=1)

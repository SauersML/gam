"""Budget descent on one Qwen3-0.6B MLP layer with intrinsic expected gates (#2951 probe, descent).

M is Qwen3-0.6B (float32). P is M with the MLP of one layer (default 14) replaced by gated slices;
every other layer, attention included, is M's. The MLP is down(silu(gate(x)) * up(x)).
Starts:
  svd     the input-whitened SVD of each of the three maps (gate and up on the MLP input x, down on
          the hidden h = silu(gate x) * up x, covariances from M's fit tokens): W C^(1/2) = U S V^T,
          slices with read C^(-1/2) v and write s u; all on = M. Each slice is gated by its own read
          r = |v.y| ||u|| (y the map's input in P), and counts one rank-one equivalent.
  neuron  neuron j of the MLP as one part: silu(g_j.x)(u_j.x) d_j, gated by its own output norm
          r_j = |silu(g_j.x)(u_j.x)| ||d_j||; all on = M. A neuron counts three rank-one equivalents
          (its gate row, up row and down column).
Thresholds start 3 s below zero (every gate on, P = M) with s = 0.1 x the root mean square of r on
M's fit tokens. Training uses the expected gate Phi((r - tau)/s), evaluation the hard gate.
Objective: KL(M || P) / A + lambda log E[k], E[k] = expected rank-one equivalents per token, A the
held-out KL of M with this layer's MLP removed (the layer's whole effect, measured at the start), and
lambda <- max(0, lambda + 0.01 log(E[k] / K)) (budget_descent.py's fixed rule, here in units of the
layer's effect, so that the step does not depend on how much the one layer matters). With
DESCENT_DUAL=anneal the step aims at a budget K_t that falls geometrically from the start's count to K
over one pass of the training windows instead of at K from the first step (a continuation from the
exact start; the fixed target cut every neuron at once).
Training: FineWeb windows of 256 tokens (qwen3_fineweb train), batch 8; held-out evaluation on the
first 8 windows of 512 tokens of the held-out shard (4096 tokens).
Usage: qwen3_layer_descent.py START K LAYER OUT.json EVAL SECONDS TRAIN_U32 HELDOUT_U32 [MODEL]"""
import sys, json, math, time, os
import numpy as np, scipy.linalg as sl, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM

start, K, layer, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
EVAL, LIMIT = int(sys.argv[5]), float(sys.argv[6])
train = np.memmap(sys.argv[7], dtype=np.uint32, mode='r').reshape(-1, 256)
heldout = np.memmap(sys.argv[8], dtype=np.uint32, mode='r').reshape(-1, 512)
name = sys.argv[9] if len(sys.argv) > 9 else 'Qwen/Qwen3-0.6B'
dev = 'cuda' if torch.cuda.is_available() else 'mps'
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
for q in model.parameters():
    q.requires_grad_(False)
mlp = model.model.layers[layer].mlp
Wg, Wu, Wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight  # [H, d], [H, d], [d, H]
ev = torch.tensor(heldout[:8].astype(np.int64), device=dev)
batch, rows = 8, train.shape[0]

state = {'mode': 'M', 'soft': [], 'hard': [], 'cache': None}
published = mlp.forward

@torch.no_grad()
def fit_inputs(ids):
    xs = []
    def grab(x):
        xs.append(x.reshape(-1, x.shape[-1]))
        return published(x)
    mlp.forward = grab
    for i in range(0, ids.shape[0], 4):
        model(ids[i:i + 4])
    mlp.forward = published
    return torch.cat(xs)

X = fit_inputs(torch.tensor(train[:32].astype(np.int64), device=dev))  # 8192 tokens
Hh = F.silu(X @ Wg.T) * (X @ Wu.T)
P = {}
def whitened_svd(W, Y):
    y = Y.cpu().double().numpy(); Wn = W.cpu().double().numpy()
    C = y.T @ y / y.shape[0]; C += 1e-6 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    lam, Q = sl.eigh(C)
    Us, S, Vt = sl.svd(Wn @ ((Q * np.sqrt(lam)) @ Q.T), full_matrices=False)
    V = torch.tensor(((Q / np.sqrt(lam)) @ Q.T) @ Vt.T, dtype=torch.float32, device=dev)
    U = torch.tensor((Us * S).T, dtype=torch.float32, device=dev)
    return V, U
with torch.no_grad():
    if start == 'svd':
        for m, W, Y in (('gate', Wg, X), ('up', Wu, X), ('down', Wd, Hh)):
            V, U = whitened_svd(W, Y)
            r = (Y @ V).abs() * U.norm(dim=1)
            s = 0.1 * r.pow(2).mean(0).sqrt().clamp_min(1e-12)
            P[m] = {'V': V.requires_grad_(), 'U': U.requires_grad_(), 'tau': (-3 * s).requires_grad_(), 's': s, 'count': 1.0}
            print(m, 'slices', V.shape[1], 'sum error', float((V @ U - W.T).abs().max()), flush=True)
    else:
        r = Hh.abs() * Wd.norm(dim=0)
        s = 0.1 * r.pow(2).mean(0).sqrt().clamp_min(1e-12)
        P['neuron'] = {'G': Wg.clone().requires_grad_(), 'Up': Wu.clone().requires_grad_(), 'D': Wd.clone().requires_grad_(),
                       'tau': (-3 * s).requires_grad_(), 's': s, 'count': 3.0}
del X, Hh
SQ2 = math.sqrt(2)

def gated(r, p):
    z = (r - p['tau']) / p['s']
    hard = (z > 0).float()
    state['hard'].append((hard * p['count']).sum(-1).reshape(-1))
    if state['mode'] == 'hard':
        return hard
    phi = 0.5 * (1 + torch.erf(z / SQ2))
    state['soft'].append((phi * p['count']).sum(-1).reshape(-1))
    return phi

def fwd(x):
    if state['mode'] == 'M':
        return published(x)
    if state['mode'] == 'zero':
        return torch.zeros_like(x)
    if start == 'svd':
        def apply(m, y):
            p = P[m]; c = y @ p['V']
            if state['mode'] == 'all':
                return c @ p['U']
            return (c * gated(c.abs() * p['U'].norm(dim=1), p)) @ p['U']
        h = F.silu(apply('gate', x)) * apply('up', x)
        return apply('down', h)
    p = P['neuron']
    h = F.silu(x @ p['G'].T) * (x @ p['Up'].T)
    if state['mode'] == 'all':
        return h @ p['D'].T
    return (h * gated(h.abs() * p['D'].norm(dim=0), p)) @ p['D'].T
mlp.forward = fwd

def kl_bits(lm, lp):
    pm = F.log_softmax(lm.float(), -1); pp = F.log_softmax(lp.float(), -1)
    return (pm.exp() * (pm - pp)).sum(-1) / math.log(2)

def run(ids, mode):
    state['mode'], state['soft'], state['hard'] = mode, [], []
    return model(ids).logits

@torch.no_grad()
def evaluate():
    r = {'kl': [], 'kl_soft': [], 'kl_all_on': [], 'active': [], 'active_soft': [], 'per_map': []}
    for i in range(0, ev.shape[0], 2):
        ids = ev[i:i + 2]
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

@torch.no_grad()
def ablated():
    return float(np.mean([kl_bits(run(ev[i:i + 2], 'M'), run(ev[i:i + 2], 'zero')).mean().item() for i in range(0, ev.shape[0], 2)]))

trainable = [p[k] for p in P.values() for k in ('V', 'U', 'G', 'Up', 'D') if k in p]
# Adam steps of 0.3% of each tensor's root mean square; thresholds 1% of their noise scale x 10.
groups = [{'params': [q], 'lr': 3e-3 * q.detach().pow(2).mean().sqrt().item()} for q in trainable]
groups += [{'params': [p['tau']], 'lr': 0.1 * p['s'].mean().item()} for p in P.values()]
opt = torch.optim.Adam(groups)
lam, rng = 0.0, np.random.default_rng(0)
A = ablated()
print('KL with the layer\'s MLP removed', A, flush=True)
log = {'model': name, 'layer': layer, 'start': start, 'K': K, 'kl_ablated': A, 'trace': []}
e = evaluate(); print('start', e, flush=True); log['trace'].append({'step': 0, **e})
t0 = time.time()
step = 0
while True:
    ids = torch.tensor(train[rng.integers(32, rows, batch)].astype(np.int64), device=dev)
    with torch.no_grad():
        lm = run(ids, 'M')
    lp = run(ids, 'soft')
    kl = kl_bits(lm, lp).mean()
    ek = torch.stack(state['soft']).sum(0).mean()
    hk = torch.stack(state['hard']).sum(0).mean().item()
    if step == 0:
        K0 = max(ek.item(), K)
    Kt = K0 * (K / K0) ** min(1.0, step * batch / rows) if os.environ.get('DESCENT_DUAL') == 'anneal' else K
    opt.zero_grad(); (kl / A + lam * torch.log(ek)).backward(); opt.step()
    lam = max(0.0, lam + 0.01 * math.log(ek.item() / Kt))
    step += 1
    last = time.time() - t0 > LIMIT
    if step % EVAL == 0 or last:
        e = evaluate()
        rec = {'step': step, 'lambda': lam, 'K_t': Kt, 'train_kl': kl.item(), 'train_k_soft': ek.item(), 'train_k_hard': hk, **e,
               'seconds': time.time() - t0}
        log['trace'].append(rec); print(rec, flush=True)
        json.dump(log, open(out, 'w'), indent=1)
    if last:
        break

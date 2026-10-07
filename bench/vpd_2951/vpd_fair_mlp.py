"""VPD's fair MLP-only bar on vpd4l (#2951, descent): held-out KL(M || P) in bits per token and the
active subcomponents per token over the 8 MLP maps, where P is M with its MLP maps replaced by VPD's
masked subcomponents (no delta) and attention exact, on rows 1024..1031 (as budget_descent.py).
Masks are VPD's causal-importance network's (clamp(CI, 0, 1)), computed four ways:
  m_clean        from M's clean inputs to all 24 sites, the network as published (bidirectional)
  m_causal       from M's clean inputs, the network's attention made causal
  own_causal_1   from P's own inputs to the sites with every MLP mask 1 (one round, reads nothing
                 of M beyond the exact attention it shares), the network made causal: autonomous
                 and causal, the fair bar
  own_1          as own_causal_1 with the published bidirectional network
Usage: vpd_fair_mlp.py GAM TARGET VPD_PTH TOKENS OUT.json"""
import sys, json, math, types
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
import vpd_model
from vpd_model import load_target, load_vpd, site_names, lower_leaky
vpd_model.TARGET_DIR = Path(sys.argv[2])
if not (vpd_model.TARGET_DIR / 'model_step_99999.safetensors').exists():  # the MATS copy holds the .pt
    vpd_model.load_file = lambda f: torch.load(f.replace('.safetensors', '.pt'), map_location='cpu', weights_only=True)
vpd_model.VPD_PTH = Path(sys.argv[3])
TOKENS, out = sys.argv[4], sys.argv[5]
dev = 'cuda' if torch.cuda.is_available() else 'mps'
T = load_target(dev); V = load_vpd(T, dev)
mlp = [n for n in site_names() if '.mlp.' in n]
tok = np.memmap(TOKENS, dtype=np.uint16 if TOKENS.endswith('.u16') else np.float64, mode='r').reshape(-1, 513)
ev = torch.tensor(tok[1024:1032, :512].astype(np.int64), device=dev)


def causal_forward(self, x):
    """CIBlock.forward with causal attention: no position reads a later one."""
    B, S, D = x.shape
    h = F.rms_norm(x, (D,))
    sp = lambda t: t.view(B, S, self.n_heads, self.dh).transpose(1, 2)
    q, k, v = sp(h @ self.wq.T), sp(h @ self.wk.T), sp(h @ self.wv.T)
    q, k = self._rope(q, S), self._rope(k, S)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = x + a.transpose(1, 2).reshape(B, S, D) @ self.wo.T
    return x + self.fc2(F.gelu(self.fc1(F.rms_norm(x, (D,)))))


published = [b.forward for b in V.ci_fn.blocks]
def ci(acts, causal):
    for b, f in zip(V.ci_fn.blocks, published):
        b.forward = types.MethodType(causal_forward, b) if causal else f
    return {n: lower_leaky(v) for n, v in V.ci_fn(acts).items()}


def inputs_of(ids, mlp_masks):
    """Inputs to all 24 sites on a run with the MLP maps masked (None: M itself)."""
    V.clear()
    for n in V.names:
        st = T.site(n); st.cache_input = True
        if mlp_masks is not None and n in mlp_masks:
            st.mask = mlp_masks[n]
    logits = T(ids)
    acts = {n: T.site(n).last_input for n in V.names}
    V.clear()
    return logits, acts


def kl_bits(lm, lp):
    pm = F.log_softmax(lm.double(), -1); pp = F.log_softmax(lp.double(), -1)
    return ((pm.exp() * (pm - pp)).sum(-1) / math.log(2)).mean().item()


res = {k: {'kl': [], 'active': [], 'per_map': []} for k in ('m_clean', 'm_causal', 'own_1', 'own_causal_1', 'all_on')}
with torch.no_grad():
    for i in range(ev.shape[0]):
        ids = ev[i:i + 1]
        lm, m_acts = inputs_of(ids, None)
        ones = {n: torch.ones(1, ids.shape[1], V.C[n], device=dev) for n in mlp}
        la, own_acts = inputs_of(ids, ones)
        settings = {'m_clean': ci(m_acts, False), 'm_causal': ci(m_acts, True), 'own_1': ci(own_acts, False),
                    'own_causal_1': ci(own_acts, True), 'all_on': None}
        for k, masks in settings.items():
            m = ones if masks is None else {n: masks[n] for n in mlp}
            lp, _ = inputs_of(ids, m)
            res[k]['kl'].append(kl_bits(lm, lp))
            counts = [(m[n] > 0).float().sum(-1).mean().item() for n in mlp]
            res[k]['active'].append(sum(counts)); res[k]['per_map'].append(counts)
        print(i, {k: round(v['kl'][-1], 3) for k, v in res.items()}, flush=True)
summary = {k: {'kl': float(np.mean(v['kl'])), 'active': float(np.mean(v['active'])),
               'per_map': [round(float(x), 2) for x in np.mean(v['per_map'], 0)]} for k, v in res.items()}
print(json.dumps(summary, indent=1))
json.dump(summary, open(out, 'w'), indent=1)

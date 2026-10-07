"""Is the priced leftover unused by M? (#2951, descent) KL(M || M') on held-out text, M' being M with
each MLP map's weight W replaced by W - R = the sum of the explanation's parts (V U)^T, the leftover R
removed from M's own weights: per layer (that layer's two maps only) and all layers at once. Small
means the parts are what M uses and R is weight M does not use on this text; large means the parts
replaced M's mechanism and R patches the difference.
usage: leftover_removed.py GAM TARGET TOKENS STATE.pt [STATE.pt ...]"""
import sys, math
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
import vpd_model
from vpd_model import load_target, site_names
vpd_model.TARGET_DIR = Path(sys.argv[2])
TOKENS = sys.argv[3]
T = load_target('cpu')
tok = np.memmap(TOKENS, dtype=np.uint16 if TOKENS.endswith('.u16') else np.float64, mode='r').reshape(-1, 513)
ev = torch.tensor(tok[1024:1032, :512].astype(np.int64))
mlp = [n for n in site_names() if '.mlp.' in n]
plain = {n: T.site(n).W.clone() for n in mlp}


@torch.no_grad()
def kl_to_M():
    """Held-out KL(M || current model) in bits per token, two sequences at a time."""
    tot = 0.0
    for i, lm in enumerate(M_logp):
        lp = F.log_softmax(T(ev[2 * i:2 * i + 2]).float(), -1)
        tot += ((lm.exp() * (lm - lp)).sum(-1) / math.log(2)).mean().item()
    return tot / len(M_logp)


with torch.no_grad():
    M_logp = [F.log_softmax(T(ev[i:i + 2]).float(), -1) for i in range(0, ev.shape[0], 2)]
for path in sys.argv[4:]:
    S = torch.load(path, map_location='cpu', weights_only=False)
    parts = {n: (S['maps'][n]['V'] @ S['maps'][n]['U']).T.float() for n in mlp}       # W - R, [d_out, d_in]
    rel = {n: ((plain[n] - parts[n]).norm() / plain[n].norm()).item() for n in mlp}
    out = {}
    for layers in [[0], [1], [2], [3], [0, 1, 2, 3]]:
        for n in mlp:
            T.site(n).W.copy_(parts[n] if int(n.split('.')[1]) in layers else plain[n])
        out['all' if len(layers) == 4 else f'layer {layers[0]}'] = round(kl_to_M(), 4)
    for n in mlp:
        T.site(n).W.copy_(plain[n])
    print(path.split('/')[-1], 'step', S['step'], 'KL(M || M without its leftover):', out, '| ||R||/||W||:', {n.replace('.mlp', ''): round(v, 3) for n, v in rel.items()}, flush=True)

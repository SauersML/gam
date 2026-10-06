"""Per-layer ReLU transcoders for vpd4l (Goodfire's 4L LlamaSimpleMLP, #2951), the form
`library_transcoder` imports: layer l's transcoder reads the MLP's input x (the second RMSNorm's
output with its gain, the library's `layer.normed`) and writes the MLP's output
y = down_proj(gelu_tanh(c_fc x)) (vpd4l's MLP has no bias) as
    y_hat = sum_i relu(g_i . x + c_i) u_i + b
(W_enc rows g_i, b_enc c_i, W_dec rows u_i, b_dec b; one safetensors file per layer).

Training, all four layers at once on activations made on the fly from the engine export's weights
(no vpd_model), on export rows >= 40000 (the Pile train split; a library fit trains on rows < 33800
and holds out rows 1024..1056; positions >= 1, since the library runs M's MLP at position 0):
  1. TopK (k = TARGET_L0) for SECONDS, minimizing the fraction of variance unexplained (FVU), with
     dead features re-seeded from high-error tokens every 1500 steps in the first 70%.
  2. The ReLU form: theta_i is feature i's (1 - f_i) quantile of its pre-activation on a calibration
     sample, f_i its TopK firing frequency there, and b_enc is lowered by theta, so each feature fires
     about as often as under TopK; the encoder is then frozen (the per-token L0 stays TopK's) and the
     decoder refit for REFIT_SECONDS.
x and y are scaled to E|x|^2 = E|y|^2 = d during training; the files hold the unscaled maps.

usage: vpd4l_transcoders.py EXPORT OUT_DIR WIDTH TARGET_L0 SECONDS REFIT_SECONDS BATCH_SEQUENCES
"""
import json, math, os, sys, time
import numpy as np, torch
from safetensors.torch import save_file

EXPORT = sys.argv[1]
out_dir, F, target, seconds = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
REFIT, B = float(sys.argv[6]), int(sys.argv[7])
os.makedirs(out_dir, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'mps'
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
cfg = json.load(open(f'{EXPORT}/export.json'))
shapes = {k: v['shape'] for k, v in cfg['files'].items()}
w = lambda k: torch.from_numpy(np.fromfile(f'{EXPORT}/{k}.f64', '<f8').reshape(shapes[k]).astype(np.float32)).to(dev)
L, d, H, hd, eps = 4, 768, 6, 128, 1e-6
P = {k: w(k) for k in shapes if k.startswith('blocks.') or k in ('wte', 'final_norm.gain')}
pos = torch.arange(512, dtype=torch.float32)
freq = 10000.0 ** (torch.arange(hd // 2, dtype=torch.float32) / (hd / 2))
ang = pos[:, None] / freq.repeat(2)[None, :]
COS, SIN = ang.cos().to(dev), ang.sin().to(dev)
tok = np.memmap(f'{EXPORT}/tokens.f64', dtype='<f8', mode='r', shape=tuple(shapes['tokens']))


def rms(x, g):
    return g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rope(x):
    n = hd // 2
    return x * COS[:x.shape[-2]] + torch.cat([-x[..., n:], x[..., :n]], -1) * SIN[:x.shape[-2]]


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


@torch.no_grad()
def activations(rows):
    ids = torch.from_numpy(np.asarray(tok[rows, :512]).astype(np.int64)).to(dev)
    Bn, Tn = ids.shape
    x = P['wte'][ids]
    xs, ys = [], []
    for i in range(L):
        g = lambda k: P[f'blocks.{i}.{k}']
        h = rms(x, g('rms1.gain'))
        q = rope((h @ g('attn.q_proj').T).view(Bn, Tn, H, hd).transpose(1, 2))
        k = rope((h @ g('attn.k_proj').T).view(Bn, Tn, H, hd).transpose(1, 2))
        v = (h @ g('attn.v_proj').T).view(Bn, Tn, H, hd).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + y.transpose(1, 2).reshape(Bn, Tn, -1) @ g('attn.o_proj').T
        hn = rms(x, g('rms2.gain'))
        m = gelu_tanh(hn @ g('mlp.c_fc').T) @ g('mlp.down_proj').T
        x = x + m
        xs.append(hn[:, 1:].reshape(-1, d)); ys.append(m[:, 1:].reshape(-1, d))
    activations.logits = rms(x, P['final_norm.gain']) @ P['wte'].T
    return xs, ys


# Scales: E|x|^2 = d and E|y|^2 = d after dividing by s_x, s_y (folded into the file exactly).
xs, ys = activations(np.arange(40000, 40016))
sx = [float(x.pow(2).sum(-1).mean().div(d).sqrt()) for x in xs]
sy = [float(y.pow(2).sum(-1).mean().div(d).sqrt()) for y in ys]
ymean = [ys[i].mean(0) / sy[i] for i in range(L)]
print('scales x', sx, 'y', sy, flush=True)


class TC(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.We = torch.nn.Parameter(torch.randn(F, d, device=dev) / math.sqrt(d))
        self.be = torch.nn.Parameter(torch.zeros(F, device=dev))
        u = torch.randn(F, d, device=dev)
        self.Wd = torch.nn.Parameter(0.1 * u / u.norm(dim=1, keepdim=True))
        self.bd = torch.nn.Parameter(ymean[i].clone())

    def forward(self, x, topk=None):
        z = x @ self.We.T + self.be
        if topk:
            v, ix = z.topk(topk, dim=-1)
            a = torch.zeros_like(z).scatter(-1, ix, torch.relu(v))
        else:
            a = torch.relu(z)
        return a, a @ self.Wd + self.bd


tcs = [TC(i) for i in range(L)]
opts = [torch.optim.Adam(t.parameters(), lr=5e-4, betas=(0.9, 0.999)) for t in tcs]
RESAMPLE = 1500
fired = [torch.zeros(F, device=dev) for _ in range(L)]
start, step, log = time.time(), 0, []
next_row = 40016
phase = 'topk'


def train_step(i, x, y, topk, lr):
    a, yh = tcs[i](x, topk)
    var = (y - y.mean(0)).pow(2).sum(-1).mean()
    fvu = (y - yh).pow(2).sum(-1).mean() / var
    for g in opts[i].param_groups:
        g['lr'] = lr
    opts[i].zero_grad(set_to_none=True)
    fvu.backward()
    opts[i].step()
    return a.detach(), yh.detach(), float(fvu)


# Phase 1: TopK (k = target) for `seconds`. Phase 2: the ReLU form the library imports, relu(z - theta)
# with theta_i set so that feature i fires as often as under TopK (its (1 - f_i) quantile of z on a
# calibration sample), encoder frozen (so the per-token L0 stays the TopK one) and the decoder refit
# for REFIT seconds.
while True:
    elapsed = time.time() - start
    if phase == 'topk' and elapsed >= seconds:
        with torch.no_grad():
            xs, ys = activations(np.arange(next_row, next_row + 32))
            next_row += 32
            for i in range(L):
                x = xs[i] / sx[i]
                z = x @ tcs[i].We.T + tcs[i].be
                v, ix = z.topk(target, dim=-1)
                f = torch.zeros(F, device=dev).index_add_(0, ix.flatten(), torch.ones(ix.numel(), device=dev) * (v.flatten() > 0)) / z.shape[0]
                zc = z.cpu()
                theta = torch.empty(F)
                for j0 in range(0, F, 512):
                    zz = zc[:, j0:j0 + 512].sort(dim=0).values
                    idx = ((1 - f[j0:j0 + 512].cpu()) * (zz.shape[0] - 1)).round().long().clamp(0, zz.shape[0] - 1)
                    theta[j0:j0 + 512] = zz.gather(0, idx.unsqueeze(0)).squeeze(0)
                theta = torch.where(f.cpu() > 0, theta.clamp(min=0), zc.max(0).values + 1.0)
                tcs[i].be -= theta.to(dev)
                tcs[i].We.requires_grad_(False)
                tcs[i].be.requires_grad_(False)
                opts[i] = torch.optim.Adam([tcs[i].Wd, tcs[i].bd], lr=5e-4)
                print(f'layer {i}: thresholds median {float(theta.median()):.3f}, features firing in TopK {int((f > 0).sum())}', flush=True)
        phase, refit_start = 'relu', time.time()
    if phase == 'relu' and time.time() - refit_start >= REFIT:
        break
    rows = np.arange(next_row, next_row + B)
    next_row += B
    xs, ys = activations(rows)
    if phase == 'topk':
        frac = elapsed / seconds
        lr = 5e-4 * (1.0 if frac < 0.8 else max(0.1, (1 - frac) / 0.2))
    else:
        frac = (time.time() - refit_start) / REFIT
        lr = 3e-4 * (1.0 if frac < 0.7 else max(0.05, (1 - frac) / 0.3))
    stats = []
    for i in range(L):
        x, y = xs[i] / sx[i], ys[i] / sy[i]
        a, yh, fvu = train_step(i, x, y, target if phase == 'topk' else None, lr)
        l0 = float((a > 0).float().sum(-1).mean())
        fired[i] += (a > 0).float().sum(0)
        stats.append((fvu, l0))
        if phase == 'topk' and step > 0 and step % RESAMPLE == 0 and frac < 0.7:
            with torch.no_grad():
                dead = (fired[i] == 0).nonzero().flatten()
                if len(dead):
                    err = (y - yh).pow(2).sum(-1)
                    pick = torch.multinomial(err, len(dead), replacement=True)
                    xe, re = x[pick], (y - yh)[pick]
                    tcs[i].We[dead] = xe / xe.norm(dim=1, keepdim=True)
                    tcs[i].be[dead] = 0.0
                    tcs[i].Wd[dead] = re / re.norm(dim=1, keepdim=True) * 0.1
                    st = opts[i].state
                    for p in (tcs[i].We, tcs[i].be, tcs[i].Wd):
                        if p in st:
                            st[p]['exp_avg'][dead] = 0
                            st[p]['exp_avg_sq'][dead] = 0
                print(f'step {step} layer {i}: resampled {len(dead)} dead features', flush=True)
            fired[i].zero_()
    if step % 100 == 0:
        print(f'{phase} step {step} t {time.time() - start:.0f}s lr {lr:.2e} ' + ' | '.join(f'L{i} fvu {f:.4f} l0 {l:.1f}' for i, (f, l) in enumerate(stats)), flush=True)
        log.append({'phase': phase, 'step': step, 'seconds': time.time() - start, 'fvu': [s[0] for s in stats], 'l0': [s[1] for s in stats]})
    step += 1

# Write each layer in circuit-tracer's layout with the scales folded in.
for i in range(L):
    t = tcs[i]
    tensors = {
        'W_enc': (t.We.detach() / sx[i]).float().cpu().contiguous(),
        'b_enc': t.be.detach().float().cpu().contiguous(),
        'W_dec': (t.Wd.detach() * sy[i]).float().cpu().contiguous(),
        'b_dec': (t.bd.detach() * sy[i]).float().cpu().contiguous(),
    }
    save_file(tensors, f'{out_dir}/layer_{i}.safetensors')
json.dump({'width': F, 'target_l0': target, 'steps': step, 'tokens': step * B * 511, 'rows_from': 40000, 'rows_to': int(next_row), 'scales_x': sx, 'scales_y': sy, 'refit_seconds': REFIT, 'log': log}, open(f'{out_dir}/train.json', 'w'))
print('done', step, 'steps', flush=True)

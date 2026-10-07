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
small causal router per layer added to the own read; router_pure, the router alone; share, learned
gate sharing across the slices of a layer (below).
Objective (DESCENT_F=1): F, the bits-back code length per training token (below); otherwise KL alone.
Exactness (DESCENT_EXACT=1): each map's slices are parametrized by a read frame F and its canonical
dual, so they sum to M's weight at every step whatever F becomes (frame(), below); training moves
only the frames and the gates, and kl_all_on stays at rounding.
Wiring (DESCENT_EDGES=1, own arm): the MLP parts' reads of the residual stream are an explicit, fitted
graph (below), each kept edge charged its index bits.
Dual step (DESCENT_DUAL): measured (default, library_mdl's rule, below); anneal, the measured rule
toward a budget K_t that falls geometrically from the start's expected count to K over one pass of
the training rows (a continuation from the start, so that an exact start is not cut down at once); or
fixed (lambda <- lambda +
0.01 log(E[k] / K) on the loss KL + lambda log E[k], the first runs' rule).
Training rows 0..1023 of tokens.f64, held-out evaluation rows 1024..1031 (4096 tokens), where VPD's
causal-importance masks give KL 0.737 at 129 active MLP slices per token.
Usage: budget_descent.py START K STEPS OUT.json GATE EVAL SECONDS GAM TARGET VPD TOKENS
  GATE     mf = expected gate forward and derivative; st = hard gate forward, expected gate derivative;
           ramp = a continuous gate clamp((r - tau)/w, 0, 1) with a learned width w per slice, as VPD's
           masks are continuous at evaluation: trained as its expectation under the threshold noise
           (scale s), evaluated as the ramp itself (not binarized), active where r > tau
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
# DESCENT_EXACT=1: every map's slices sum to M's weight at every step by construction (below), so
# matmuls run in full float32 (TF32's 10-bit mantissa would break the sum).
EXACT = os.environ.get('DESCENT_EXACT') == '1'
torch.backends.cuda.matmul.allow_tf32 = not EXACT
torch.manual_seed(0)
T = load_target(dev)
mlp = [n for n in site_names() if '.mlp.' in n]
# DESCENT_SITES=all: the attention maps are explained too (below; always exact by construction).
attn = [n for n in site_names() if '.attn.' in n] if os.environ.get('DESCENT_SITES') == 'all' else []
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
    xs = {n: [] for n in mlp + attn}
    for i in range(0, ids.shape[0], 4):
        for n in mlp + attn: T.site(n).cache_input = True
        T(ids[i:i + 4])
        for n in mlp + attn:
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
def frame(F, W):
    """Exact slices of W from a read frame F (C x d_in rows f_i, C >= d_in, full rank): reads the
    canonical dual g_i = (F^T F)^-1 f_i and writes W f_i, so sum_i (W f_i) g_i^T = W F^T F (F^T F)^-1 = W
    for every F. Through the thin QR F = Q R: G^T = R^-1 Q^T, whose rounding error grows with cond(F),
    not cond(F)^2. Returns V (d_in x C, the reads as columns) and U (C x d_out, the writes)."""
    Q, R = torch.linalg.qr(F)
    return torch.linalg.solve_triangular(R, Q.T, upper=True), F @ W.T

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
    if EXACT:
        # The start's reads kept (the canonical dual is an involution: the frame R (R^T R)^-1 has dual
        # R), its writes replaced by the exact ones. The SVD start's down_proj has fewer slices than
        # inputs, so its frame is the hidden (neuron) axis: f_i = e_i, slice i = neuron i.
        if start == 'svd' and n.endswith('down_proj'):
            F0 = torch.eye(W.shape[1], device=dev)
        else:
            Rd = V.T.cpu().double().numpy()
            F0 = torch.tensor(Rd @ np.linalg.inv(Rd.T @ Rd), dtype=torch.float32, device=dev)
        with torch.no_grad():
            V, U = frame(F0, W)
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
    if EXACT:
        P[n]['F'] = F0.clone().requires_grad_()
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
# DESCENT_ARM=share: parts of any rank across a layer's two maps, one gate each, the grouping learned
# in both directions. Part m of layer l is gated at its input side by the norm of its c_fc members'
# own reads, sqrt(sum_j r_j^2) > t_m (rotation-invariant within the part); each down_proj slice
# fires with the part it belongs to, or alone on its own read of the hidden vector. Every slice keeps
# 8 candidate parts: a c_fc slice its own part (it starts alone in it) and the 7 c_fc parts whose
# firing on M's fit tokens is nearest its own (Hamming distance); a down_proj slice its own read and
# the 7 nearest c_fc parts. The assignment is a softmax over the candidates (logit 6 at the start's
# choice, 0 elsewhere, so a slice can join a part or leave one alike) in training and its argmax at
# evaluation. At the start every c_fc slice is alone and every down_proj slice is on its own read,
# so the start is the own-read start and no grouping is fixed; parts form only where training moves
# slices together. k counts slices (rank-one equivalents) whatever their part.
SHARE = {}
if ARM == 'share':
    with torch.no_grad():
        for l in range(4):
            fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
            on = lambda m: (((X[m] @ P[m]['V']).abs() * P[m]['U'].norm(dim=1)) > P[m]['tau']).float()
            gf, gd = on(fc), on(dn)
            ff = gf.sum(0)[:, None] + gf.sum(0)[None, :] - 2 * (gf.T @ gf)
            ff.fill_diagonal_(-1.0)
            cand_fc = ff.topk(8, dim=1, largest=False).indices            # own part first
            fd = gf.sum(0)[:, None] + gd.sum(0)[None, :] - 2 * (gf.T @ gd)
            near = fd.topk(7, dim=0, largest=False).indices.T             # [Cd, 7] c_fc parts
            cand_dn = torch.cat([torch.full((gd.shape[1], 1), -1, device=dev, dtype=torch.long), near], 1)
            L_fc = torch.zeros(gf.shape[1], 8, device=dev); L_fc[:, 0] = 6.0
            L_dn = torch.zeros(gd.shape[1], 8, device=dev); L_dn[:, 0] = 6.0
            SHARE[l] = {'cand_fc': cand_fc, 'cand_dn': cand_dn, 'L_fc': L_fc.requires_grad_(), 'L_dn': L_dn.requires_grad_(),
                        't': P[fc]['tau'].detach().clone().requires_grad_(), 's': P[fc]['s'].clone(), 'fc': fc, 'dn': dn}
# DESCENT_EDGES=1: explicit wiring between parts. A c_fc slice B of layer l (a part that reads the
# residual stream) reads x_B = embedding + every attention output (exact, as in M) + the outputs of the
# earlier layers' down_proj slices A (the parts that write the stream) on its input list in(B), not the
# whole stream: its read is v_B . (gain_l * x_B) / r, the scale r = RMS of P's own full stream at the
# pre-MLP norm (all executed parts, so P stays autonomous) and gain_l the norm's gain. Writing a_A for
# A's coefficient on the token (its read times its gate) and u_A for its write,
#   read_B = v_B . h - sum_{A not in in(B)} ((gain_l * v_B) . u_A) a_A / r,
# h the full normed stream, so with every edge on the read is the slice's own (P unchanged). The edge
# A -> B is kept where eta_AB > 0: in training the expected gate Phi(eta_AB) under unit Gaussian noise,
# at evaluation the hard gate 1[eta_AB > 0]. Every kept edge costs its index bits, log2 of the number
# of candidate sources of B (every down_proj slice of the earlier layers), added to the objective per
# training token (/ N); the start keeps every edge (eta = 3), which is exact, and F prunes. An edge is
# executed machinery on a token where it is kept and both its parts are on, so the budget counts it:
# k = parts on + edges on per token (in training the expectation, Phi(eta) times both parts' expected
# gates; at evaluation the hard count).
EDGES = os.environ.get('DESCENT_EDGES') == '1'
EDGE = {}
if EDGES:
    if ARM != 'own' or start == 'vpdgroup':
        raise SystemExit('DESCENT_EDGES: the own arm with per-slice gates only (a shared or separate gate reads the full stream)')
    for l in range(1, 4):
        n_w = sum(P[f'h.{k}.mlp.down_proj']['V'].shape[1] for k in range(l))
        EDGE[l] = {'eta': torch.full((n_w, P[f'h.{l}.mlp.c_fc']['V'].shape[1]), 3.0, device=dev, requires_grad=True), 'bits': math.log2(n_w)}
        print(f'layer {l}: {n_w} x {EDGE[l]["eta"].shape[1]} candidate edges, {EDGE[l]["bits"]:.2f} bits each', flush=True)
    # The pre-MLP norm's scale r of P's own full stream, recorded per layer as the model computes it.
    MLP_NORM = {id(T.norms[2 * l + 1]): l for l in range(4)}
    plain_rms = vpd_model.rms
    def recording_rms(x, w, eps):
        if id(w) in MLP_NORM:
            state['r'][MLP_NORM[id(w)]] = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
        return plain_rms(x, w, eps)
    vpd_model.rms = recording_rms
# DESCENT_SITES=all, attention: each head h of each attention map is its own block of slices, exact
# by construction on the head's 128-dimensional side. A frame F_h (C x 128, rows f_i, full rank):
#   q, k, v (the head's output side): slice i writes f_i into the head's coordinates and reads
#     W_h^T g_i, g_i the canonical dual of the frame, so sum_i f_i (g_i . W_h x) = W_h x;
#   o (the head's input side, its 128-dimensional output a_h): slice i reads g_i . a_h and writes
#     W_h f_i, so sum_i (W_h f_i)(g_i . a_h) = W_h a_h,
# for any F, through the batched thin QR as frame() does. Each slice is gated by its own read
# r = |coefficient| ||write|| on its own input at its own position (causal), and counts one rank-one
# equivalent. VPD start: q, k, v keep VPD's writes restricted to the head as the frame; o keeps VPD's
# reads restricted to the head (the frame R (R^T R)^-1 has dual R); thresholds one per map, at the
# quantile of r matching VPD's mean active count at that map (VPD_ATTN_COUNTS, from its masks on M's
# inputs, rows 1024..1031, vpd_fair_mlp.py with FAIR_SITES=all).
VPD_ATTN_COUNTS = {'h.0.attn.q_proj': 0.91, 'h.0.attn.k_proj': 1.25, 'h.0.attn.v_proj': 1.9, 'h.0.attn.o_proj': 2.47,
                   'h.1.attn.q_proj': 1.0, 'h.1.attn.k_proj': 1.22, 'h.1.attn.v_proj': 3.6, 'h.1.attn.o_proj': 5.07,
                   'h.2.attn.q_proj': 4.28, 'h.2.attn.k_proj': 4.22, 'h.2.attn.v_proj': 10.16, 'h.2.attn.o_proj': 15.76,
                   'h.3.attn.q_proj': 1.99, 'h.3.attn.k_proj': 2.04, 'h.3.attn.v_proj': 7.74, 'h.3.attn.o_proj': 12.9}
NH, HD = T.n_head, T.hd


def head_frame(F, W, o):
    """Reads V [H, d_in_h, C] and writes U [H, C, d_out_h] of the head blocks of W from frames F [H, C, HD]."""
    Q, R = torch.linalg.qr(F)
    Gt = torch.linalg.solve_triangular(R, Q.transpose(1, 2), upper=True)       # [H, HD, C], the dual^T
    if o:
        Wh = W.view(W.shape[0], NH, HD).permute(1, 0, 2)                        # [H, d_model, HD]
        return Gt, F @ Wh.transpose(1, 2)
    Wh = W.view(NH, HD, W.shape[1])                                              # [H, HD, d_model]
    return Wh.transpose(1, 2) @ Gt, F


def head_coefficients(x, V, o):
    """Every slice's coefficient [H, tokens, C] on the map's input x [tokens, d_in]."""
    return (x.view(-1, NH, HD).permute(1, 0, 2) @ V) if o else (x @ V)


def head_output(c, U, o):
    """The map's output [tokens, d_out] from the gated coefficients c [H, tokens, C]."""
    y = c @ U
    return y.sum(0) if o else y.permute(1, 0, 2).reshape(c.shape[1], -1)


A = {}
for n in attn:
    if start != 'vpd':
        raise SystemExit('DESCENT_SITES=all: the vpd start only')
    W = T.site(n).W; o = n.endswith('o_proj')
    Vv, Uv = load(n + '.V'), load(n + '.U')                                      # [768, C], [C, 768]
    if o:
        Rh = Vv.view(NH, HD, -1).transpose(1, 2).cpu().double()                  # [H, C, HD] reads
        F0 = (Rh @ torch.linalg.inv(Rh.transpose(1, 2) @ Rh)).float().to(dev)
    else:
        F0 = Uv.view(-1, NH, HD).permute(1, 0, 2).contiguous()                  # [H, C, HD] writes
    with torch.no_grad():
        V, U = head_frame(F0, W, o)
        x = X[n]
        c = head_coefficients(x, V, o)
        r = c.abs() * U.norm(dim=-1)[:, None, :]
        s_ = 0.1 * r.pow(2).mean(1).sqrt().clamp_min(1e-12)                     # [H, C]
        q = 1 - VPD_ATTN_COUNTS.get(n, 1.0) / (NH * V.shape[-1])
        flat = r.reshape(-1); idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
        tau = torch.full_like(s_, torch.quantile(flat[idx], q).item())
        err = (head_output(c, U, o) - x @ W.T).abs().max().item()
        sv = torch.linalg.svdvals(F0.cpu().double())
        cond = (sv[:, 0] / sv[:, -1]).max().item()
    A[n] = {'F': F0.clone().requires_grad_(), 'V': V, 'U': U, 'tau': tau.clone().requires_grad_(), 's': s_, 'o': o}
    print(n, 'slices', NH * V.shape[-1], 'all-on error', err, 'worst head frame cond', round(cond, 1), flush=True)
del X

state = {'mode': 'M', 'soft': [], 'hard': [], 'gates': {}, 'route': {}, 'share': {}, 'r': {}, 'writers': {}, 'on': {}, 'gate': {}, 'edges_soft': [], 'edges_hard': []}
SQ2 = math.sqrt(2)
def make(n):
    st = T.site(n); p = P[n]; grp = GROUP.get(n); mult = MULT.get(n)
    layer = int(n.split('.')[1]); router = ROUTER.get(layer)
    if grp is not None:
        tied, own = grp[1] >= 0, grp[1].clamp_min(0)
    def emit(coef):
        # A down_proj slice's coefficient on each token (its read times its gate): what it writes.
        if n.endswith('down_proj'):
            state['writers'][layer] = coef
        return coef @ p['U']
    def fwd(x):
        if state['mode'] == 'M':
            return x @ st.W.T
        c = x @ p['V']
        if state['mode'] == 'all':
            return emit(c)
        keep = None
        if layer in EDGE and n.endswith('c_fc'):
            # The read of the listed inputs only: the full read less the edges that are off.
            E = EDGE[layer]
            keep = (E['eta'] > 0).float() if state['mode'] == 'hard' else 0.5 * (1 + torch.erf(E['eta'] / SQ2))
            writes = torch.cat([P[f'h.{k}.mlp.down_proj']['U'] for k in range(layer)])
            interaction = writes @ (T.norms[2 * layer + 1][:, None] * p['V'])
            a = torch.cat([state['writers'][k] for k in range(layer)], -1)
            c = c - (a @ ((1 - keep) * interaction)) / state['r'][layer]
        if ARM == 'share':
            S = SHARE[layer]
            r = c.abs() * p['U'].norm(dim=1)
            if n.endswith('c_fc'):
                A = torch.softmax(S['L_fc'], 1)
                R2 = torch.zeros_like(r)
                for k in range(8):
                    R2 = R2.index_add(-1, S['cand_fc'][:, k], r.pow(2) * A[:, k])
                zp = ((R2 + 1e-12).sqrt() - S['t']) / S['s']
                hp, pp = (zp > 0).float(), 0.5 * (1 + torch.erf(zp / SQ2))
                state['share'][layer] = (hp, pp)
                hard = hp[..., S['cand_fc'].gather(1, A.argmax(1, keepdim=True)).squeeze(1)]
                soft = (pp[..., S['cand_fc']] * A).sum(-1)
                if state.get('collect') is not None and state['mode'] == 'hard':
                    state['collect'].setdefault(layer, []).append((hp.reshape(-1, hp.shape[-1]) > 0, (c * hard).reshape(-1, c.shape[-1])))
            else:
                A = torch.softmax(S['L_dn'], 1)
                hp, pp = state['share'][layer]
                zo = (r - p['tau']) / p['s']
                ho, po = (zo > 0).float(), 0.5 * (1 + torch.erf(zo / SQ2))
                idx = S['cand_dn'].clamp_min(0)
                choice = A.argmax(1)
                hard = torch.where(choice == 0, ho, hp[..., idx.gather(1, choice[:, None]).squeeze(1)])
                vals = pp[..., idx]                                         # [..., Cd, 8]
                vals = torch.cat([po.unsqueeze(-1), vals[..., 1:]], -1)
                soft = (vals * A).sum(-1)
            state['hard'].append(hard.sum(-1).reshape(-1))
            if state['mode'] == 'hard':
                return emit(c * hard)
            state['soft'].append(soft.sum(-1).reshape(-1))
            return emit(c * soft)
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
        if EDGES:
            # The slice's gate (hard, expected) for the edges into later layers, and the edges on per
            # token into this layer's readers: kept, with both parts on.
            state['gate'][n] = (hard, phi)
            if keep is not None:
                with torch.no_grad():
                    hw = torch.cat([state['gate'][f'h.{k}.mlp.down_proj'][0] for k in range(layer)], -1)
                    state['edges_hard'].append(((hw @ (E['eta'] > 0).float()) * hard).sum(-1).reshape(-1))
                if state['mode'] != 'hard':
                    pw = torch.cat([state['gate'][f'h.{k}.mlp.down_proj'][1] for k in range(layer)], -1)
                    state['edges_soft'].append(((pw @ keep) * phi).sum(-1).reshape(-1))
        if gate == 'ramp':
            a, width = read - p['tau'], p['lw'].exp()
            if state['mode'] == 'hard':
                state['on'][n] = (a > 0).float()
                return emit(c * (a / width).clamp(0, 1))
            state['soft'].append((phi * w).sum(-1).reshape(-1))
            # E[clamp((a + s e)/width, 0, 1)] for e ~ N(0, 1): (g(a) - g(a - width)) / width with
            # g(y) = E[(y + s e)^+] = y Phi(y/s) + s phi(y/s).
            def g(y):
                t = y / p['s']
                return y * 0.5 * (1 + torch.erf(t / SQ2)) + p['s'] * torch.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
            return emit(c * (g(a) - g(a - width)) / width)
        if state['mode'] == 'hard':
            state['on'][n] = hard
            return emit(c * hard)
        state['soft'].append((phi * w).sum(-1).reshape(-1))
        g = phi if gate == 'mf' else hard + phi - phi.detach()
        return emit(c * g)
    return fwd
for n in mlp: T.site(n)._forward = make(n)


def make_attn(n):
    st = T.site(n); p = A[n]
    def fwd(x):
        if state['mode'] == 'M':
            return x @ st.W.T
        xin = x.reshape(-1, x.shape[-1])
        c = head_coefficients(xin, p['V'], p['o'])
        if state['mode'] == 'all':
            g = 1.0
        else:
            z = (c.abs() * p['U'].norm(dim=-1)[:, None, :] - p['tau'][:, None, :]) / p['s'][:, None, :]
            hard = (z > 0).float()
            state['hard'].append(hard.sum((0, 2)))
            if state['mode'] == 'hard':
                g = hard
            else:
                phi = 0.5 * (1 + torch.erf(z / SQ2))
                state['soft'].append(phi.sum((0, 2)))
                g = phi if gate == 'mf' else hard + phi - phi.detach()
        return head_output(c * g, p['U'], p['o']).view(x.shape[0], x.shape[1], -1)
    return fwd
for n in attn: T.site(n)._forward = make_attn(n)

def kl_bits(lm, lp):
    pm = F.log_softmax(lm.float(), -1); pp = F.log_softmax(lp.float(), -1)
    return (pm.exp() * (pm - pp)).sum(-1) / math.log(2)

def run(ids, mode):
    state['mode'], state['soft'], state['hard'], state['edges_soft'], state['edges_hard'] = mode, [], [], [], []
    return T(ids)

def example_graph(position):
    """The explanation's graph on one token (held-out row 1024 at `position`) from the last hard run:
    the active parts of every map (slices whose hard gate is on) and the kept edges between active
    parts, each with the term it adds to its reader's read, ((gain * v_B) . u_A) a_A / r."""
    active = {n: torch.nonzero(state['on'][n][0, position] > 0).squeeze(1).tolist() for n in mlp}
    edges = []
    for l, E in EDGE.items():
        fc = f'h.{l}.mlp.c_fc'
        sources = [(f'h.{k}.mlp.down_proj', j) for k in range(l) for j in range(P[f'h.{k}.mlp.down_proj']['V'].shape[1])]
        on_w = torch.cat([state['on'][f'h.{k}.mlp.down_proj'][0, position] for k in range(l)]) > 0
        a = torch.cat([state['writers'][k][0, position] for k in range(l)])
        writes = torch.cat([P[f'h.{k}.mlp.down_proj']['U'] for k in range(l)])
        interaction = writes @ (T.norms[2 * l + 1][:, None] * P[fc]['V'])
        for A in torch.nonzero(on_w).squeeze(1).tolist():
            for B in active[fc]:
                if E['eta'][A, B] > 0:
                    edges.append([*sources[A], fc, B, round((interaction[A, B] * a[A] / state['r'][l][0, position, 0]).item(), 5)])
    return {'row': 1024, 'position': position, 'active': active, 'edges': edges}

def contexts(Z):
    """Distinct firing contexts of a part: its firing tokens' member reads Z (tokens x members)
    clustered by k-means for k = 1..6, k chosen by BIC under a spherical Gaussian per cluster."""
    n, d = Z.shape
    best = (float('inf'), 1)
    for k in range(1, 7):
        if n < 10 * k:
            break
        C = Z[torch.randperm(n, device=Z.device)[:k]]
        for _ in range(20):
            a = torch.cdist(Z, C).argmin(1)
            C = torch.stack([Z[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        sse = (Z - C[torch.cdist(Z, C).argmin(1)]).pow(2).sum().item()
        bic = n * d * math.log(max(sse, 1e-30) / (n * d)) + k * (d + 1) * math.log(n)
        best = min(best, (bic, k))
    return best[1]

def share_parts(l, S):
    """Layer l's parts under the hardened assignment, from the held-out evaluation's hard run: each
    part's rank (slices of both maps), the share of tokens it fires on, and (for the 10 largest) its
    distinct firing contexts (parts with two or more c_fc members; a single read has no context
    structure to cluster)."""
    Cf = S['L_fc'].shape[0]
    fc_part = S['cand_fc'].gather(1, S['L_fc'].argmax(1, keepdim=True)).squeeze(1)
    ch = S['L_dn'].argmax(1)
    dn_part = torch.where(ch == 0, torch.full_like(ch, -1), S['cand_dn'].gather(1, ch[:, None]).squeeze(1))
    rank = torch.bincount(fc_part, minlength=Cf) + torch.bincount(dn_part[dn_part >= 0], minlength=Cf)
    has_fc = torch.bincount(fc_part, minlength=Cf) > 0
    used = torch.nonzero(has_fc).squeeze(1)
    fire = torch.cat([f for f, _ in state['collect'][l]]).float()
    reads = torch.cat([r for _, r in state['collect'][l]])
    share = fire.mean(0)
    rk = rank[used]
    out = {'parts': int(used.numel()), 'down_on_own_read': int((dn_part < 0).sum()),
           'rank_hist': {'1': int((rk == 1).sum()), '2-3': int(((rk >= 2) & (rk <= 3)).sum()), '4-7': int(((rk >= 4) & (rk <= 7)).sum()),
                         '8-15': int(((rk >= 8) & (rk <= 15)).sum()), '16+': int((rk >= 16).sum())},
           'mean_rank': round(rk.float().mean().item(), 2), 'max_rank': int(rk.max().item()),
           'mean_fire_share': round(share[used].mean().item(), 4), 'largest': []}
    for m in used[rk.argsort(descending=True)[:10]].tolist():
        members = torch.nonzero(fc_part == m).squeeze(1)
        Z = reads[fire[:, m] > 0][:, members]
        out['largest'].append({'rank': int(rank[m]), 'fc': int(members.numel()), 'fire_share': round(share[m].item(), 4),
                               'contexts': contexts(Z) if members.numel() >= 2 and Z.shape[0] >= 10 else None})
    return out

@torch.no_grad()
def evaluate(final=False):
    if EXACT:
        for n in mlp:
            P[n]['V'], P[n]['U'] = frame(P[n]['F'], T.site(n).W)
    for n in attn:
        A[n]['V'], A[n]['U'] = head_frame(A[n]['F'], T.site(n).W, A[n]['o'])
    state['collect'] = {} if SHARE else None
    r = {'kl': [], 'kl_soft': [], 'kl_all_on': [], 'active': [], 'active_soft': [], 'per_map': [], 'edges_on': [], 'edges_on_soft': []}
    graph = None
    for i in range(0, ev.shape[0], 4):
        ids = ev[i:i + 4]
        lm = run(ids, 'M')
        lp = run(ids, 'hard'); r['kl'].append(kl_bits(lm, lp).mean().item())
        r['active'].append(torch.stack(state['hard']).sum(0).mean().item())
        r['per_map'].append([h.mean().item() for h in state['hard']])
        if EDGES:
            r['edges_on'].append(torch.stack(state['edges_hard']).sum(0).mean().item())
            if final and i == 0:
                graph = example_graph(255)
        lp = run(ids, 'soft'); r['kl_soft'].append(kl_bits(lm, lp).mean().item())
        r['active_soft'].append(torch.stack(state['soft']).sum(0).mean().item())
        if EDGES:
            r['edges_on_soft'].append(torch.stack(state['edges_soft']).sum(0).mean().item())
        lp = run(ids, 'all'); r['kl_all_on'].append(kl_bits(lm, lp).mean().item())
    out = {k: float(np.mean(v)) for k, v in r.items() if k != 'per_map' and v}
    out['per_map'] = [round(float(x), 2) for x in np.mean(r['per_map'], 0)]
    if EDGES:
        # Kept edges per layer, and the distribution of each reader part's inputs (its kept edges).
        out['edges'] = {}
        for l, E in EDGE.items():
            inputs = (E['eta'] > 0).float().sum(0)
            q = torch.quantile(inputs, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0], device=dev)).tolist()
            out['edges'][l] = {'kept': int(inputs.sum().item()), 'candidates': E['eta'].numel(), 'inputs_per_part_quantiles_0_25_50_75_90_100': [round(x, 1) for x in q],
                               'inputs_per_part_mean': round(inputs.mean().item(), 2), 'parts_with_no_input': int((inputs == 0).sum().item())}
        out['edge_bits'] = sum(E['bits'] * (E['eta'] > 0).float().sum().item() for E in EDGE.values())
        if graph is not None:
            out['example_graph'] = graph
    if SHARE:
        out['parts'] = [share_parts(l, S) for l, S in SHARE.items()]
    state['collect'] = None
    return out

# Every trained tensor as (container, key, scale): Adam steps of 0.3% of the scale per step, the
# scale a weight tensor's root mean square (a router G2 started at zero: its map's noise scale) and a
# threshold's its map's noise scale x 33. DESCENT_LR multiplies every step size (default 1).
LR = float(os.environ.get('DESCENT_LR', '1'))
rms = lambda q: q.detach().pow(2).mean().sqrt().item()
slots = [(P[n], w, rms(P[n][w])) for n in mlp for w in (('F',) if EXACT else ('V', 'U')) + (('G',) if ARM == 'dir' else ())]
slots += [(A[n], 'F', rms(A[n]['F'])) for n in attn] + [(A[n], 'tau', 100 / 3 * A[n]['s'].mean().item()) for n in attn]
slots += [(P[n], 'tau', 100 / 3 * P[n]['s'].mean().item()) for n in mlp]
# The ramp's log width, started at log s (a ramp as wide as the threshold noise), by 1% per step.
if gate == 'ramp':
    for n in mlp:
        P[n]['lw'] = P[n]['s'].log().clone().requires_grad_()
    slots += [(P[n], 'lw', 10 / 3) for n in mlp]
for l, S in SHARE.items():
    # part thresholds by 1% of their noise scale x 10, assignment logits by 0.02 per step
    slots += [(S, 't', 100 / 3 * S['s'].mean().item()), (S, 'L_fc', 20 / 3), (S, 'L_dn', 20 / 3)]
for l, R in ROUTER.items():
    slots.append((R, 'G1', rms(R['G1'])))
    slots += [(R, n, rms(R[n]) or P[n]['s'].mean().item()) for n in R if n != 'G1']
# DESCENT_F=1: the objective is F, the bits-back code length per training token: E_q[KL(M || P)] on
# the training tokens plus KL(q || p) / N, N the training tokens (train_rows x 512). q is a Gaussian
# per entry, N(mu, sigma^2), mu started at the tensor and sigma at 1% of its scale; p is a Gaussian
# N(0, v) per tensor with v at its optimum mean(mu^2 + sigma^2), so KL(q || p) = (n ln v - sum ln
# sigma^2) / 2 per tensor of n entries. Each step draws one sample of every tensor; evaluation is
# at the posterior mean.
FMODE = os.environ.get('DESCENT_F') == '1'
N = train_rows * 512
leaves = []
if FMODE:
    groups = []
    for cont, key, scale in slots:
        mu = cont[key].detach().clone().requires_grad_()
        ls = torch.full_like(mu, math.log(0.01 * scale)).requires_grad_()
        leaves.append((cont, key, mu, ls))
        # log sigma by 1% per step
        groups += [{'params': [mu], 'lr': LR * 3e-3 * scale}, {'params': [ls], 'lr': LR * 1e-2}]
else:
    groups = [{'params': [cont[key]], 'lr': LR * 3e-3 * scale} for cont, key, scale in slots]
# Edge gates by 0.01 per step (from the start's 3, 300 steps to drop an edge the data never defends).
groups += [{'params': [E['eta']], 'lr': LR * 1e-2} for E in EDGE.values()]
opt = torch.optim.Adam(groups)
trainable = [q for g in groups for q in g['params']]

def draw(mean):
    """Install the posterior mean (mean) or one sample of every tensor (F only), then (exact) every
    map's reads and writes from its frame."""
    for cont, key, mu, ls in leaves:
        cont[key] = mu if mean else mu + ls.exp() * torch.randn_like(mu)
    if EXACT:
        for n in mlp:
            P[n]['V'], P[n]['U'] = frame(P[n]['F'], T.site(n).W)
    for n in attn:
        A[n]['V'], A[n]['U'] = head_frame(A[n]['F'], T.site(n).W, A[n]['o'])

def description_bits():
    """KL(q || p) in bits (F only)."""
    total = 0.0
    for _, _, mu, ls in leaves:
        v = (mu.pow(2) + (2 * ls).exp()).mean()
        total = total + 0.5 * (mu.numel() * torch.log(v) - 2 * ls.sum())
    return total / math.log(2)
# The budget's multiplier (library_mdl's rule, Settings::budget): the step descends KL + lam (E[k] - K),
# then lam <- max(0, lam + eta (E[k] - K)) with eta = lam_hat / (K B), lam_hat = |<g_F, g_k>| / |g_k|^2
# the multiplier at which the budget's gradient cancels the KL gradient's component along
# g_k = dE[k]/dtheta (both measured on this step), B the batches in one pass over the training rows;
# lam starts at the first step's lam_hat (the balance value), so the budget term begins at the strength
# that holds E[k] where it is.
B = train_rows * 512 / (batch * seq)
DUAL = os.environ.get('DESCENT_DUAL', 'measured')
lam, rng = 0.0, np.random.default_rng(0)
log = {'start': start, 'K': K, 'steps': steps, 'gate': gate, 'arm': ARM, 'dual': DUAL, 'train_rows': train_rows, 'F': FMODE, 'edges': EDGES, 'trace': []}
draw(True)
e = evaluate(); print('start', e, flush=True); log['trace'].append({'step': 0, **e})
t0 = time.time()
for step in range(steps):
    rows = rng.integers(0, train_rows, batch); rows = np.where(rows >= 1024, rows + 8, rows); offs = rng.integers(0, 513 - seq, batch)
    ids = torch.tensor(np.stack([tok[r, o:o + seq] for r, o in zip(rows, offs)]).astype(np.int64), device=dev)
    with torch.no_grad():
        lm = run(ids, 'M')
    draw(False)
    lp = run(ids, 'soft')
    kl = kl_bits(lm, lp).mean()
    # The objective's data and description terms (F; without DESCENT_F only the data term).
    desc = description_bits() if FMODE else torch.zeros((), device=dev)
    # Each kept edge's index bits, at the expected gates.
    edge_bits = sum(E['bits'] * (0.5 * (1 + torch.erf(E['eta'] / SQ2))).sum() for E in EDGE.values()) if EDGES else torch.zeros((), device=dev)
    objective = kl + (desc + edge_bits) / N
    ek = torch.stack(state['soft']).sum(0).mean()
    hk = torch.stack(state['hard']).sum(0).mean().item()
    if EDGES:
        # The per-token budget counts the edges on beside the parts on.
        ek = ek + torch.stack(state['edges_soft']).sum(0).mean()
        hk += torch.stack(state['edges_hard']).sum(0).mean().item()
    if DUAL == 'anneal':
        if step == 0:
            K0 = max(ek.item(), K)
        Kt = K0 * (K / K0) ** min(1.0, step / B)
    else:
        Kt = K
    if DUAL == 'fixed':
        opt.zero_grad(); (objective + lam * torch.log(ek)).backward(); opt.step()
        lam = max(0.0, lam + 0.01 * math.log(ek.item() / K))
        g_f = None
    else:
        g_f = torch.autograd.grad(objective, trainable, retain_graph=True, allow_unused=True)
    if g_f is not None:
        g_k = torch.autograd.grad(ek, trainable, allow_unused=True)
        along = sum((a * b).sum() for a, b in zip(g_f, g_k) if a is not None and b is not None).item()
        square = sum(b.pow(2).sum() for b in g_k if b is not None).item()
        if step == 0 and square > 0:
            lam = abs(along) / square
        for q, a, b in zip(trainable, g_f, g_k):
            q.grad = (torch.zeros_like(q) if a is None else a) + (0 if b is None else lam * b)
        opt.step()
        if square > 0:
            lam = max(0.0, lam + abs(along) / square / (Kt * B) * (ek.item() - Kt))
    last = step == steps - 1 or time.time() - t0 > LIMIT
    if (step + 1) % EVAL == 0 or last:
        draw(True)
        e = evaluate(final=last)
        rec = {'step': step + 1, 'lambda': lam, 'K_t': Kt, 'train_kl': kl.item(), 'description_bits': desc.item(), 'train_edge_bits': float(edge_bits),
               'train_F': objective.item(), 'train_k_soft': ek.item(), 'train_k_hard': hk, **e,
               'seconds': time.time() - t0}
        log['trace'].append(rec); print(rec, flush=True)
        json.dump(log, open(out, 'w'), indent=1)
    if last:
        break

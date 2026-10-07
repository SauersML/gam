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
# DESCENT_EXACT=1: every map's slices sum to M's weight at every step by construction (below). The
# frames' duals and writes are computed in full float32 (TF32's 10-bit mantissa would break the sum);
# the forward passes of M and P run in TF32 alike. DESCENT_DOWN=neuron puts every down_proj map on
# the hidden (neuron) axis, f_i = e_i, fixed: slice i is neuron i with read e_i and write W[:, i], exact
# with no dual to compute (the down maps' 3,072-wide QR dominated each exact step).
EXACT = os.environ.get('DESCENT_EXACT') == '1'
NEURON_DOWN = os.environ.get('DESCENT_DOWN') == 'neuron'
torch.backends.cuda.matmul.allow_tf32 = True


class full_float32:
    """Matmuls in full float32 inside the block."""
    def __enter__(self):
        self.was = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.was
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
# The VPD start's thresholds match VPD's per-map counts scaled down to the budget when the budget is
# below VPD's total (START_SCALE, set once both tables exist): a start above the budget made the dual
# step push every threshold up at once, and gates pushed far below threshold get no gradient back
# (the whole model from 237 active at K = 64 and 128 ended at 10-15 active, unable to recover).
VPD_COUNTS = {'h.0.mlp.c_fc': 18.62, 'h.0.mlp.down_proj': 20.20, 'h.1.mlp.c_fc': 4.23,
              'h.1.mlp.down_proj': 3.20, 'h.2.mlp.c_fc': 7.16, 'h.2.mlp.down_proj': 7.77,
              'h.3.mlp.c_fc': 26.43, 'h.3.mlp.down_proj': 41.07}
VPD_ATTN_COUNTS = {'h.0.attn.q_proj': 0.91, 'h.0.attn.k_proj': 1.25, 'h.0.attn.v_proj': 1.9, 'h.0.attn.o_proj': 2.47,
                   'h.1.attn.q_proj': 1.0, 'h.1.attn.k_proj': 1.22, 'h.1.attn.v_proj': 3.6, 'h.1.attn.o_proj': 5.07,
                   'h.2.attn.q_proj': 4.28, 'h.2.attn.k_proj': 4.22, 'h.2.attn.v_proj': 10.16, 'h.2.attn.o_proj': 15.76,
                   'h.3.attn.q_proj': 1.99, 'h.3.attn.k_proj': 2.04, 'h.3.attn.v_proj': 7.74, 'h.3.attn.o_proj': 12.9}
START_SCALE = min(1.0, K / (sum(VPD_COUNTS.values()) + (sum(VPD_ATTN_COUNTS.values()) if attn else 0.0)))

@torch.no_grad()
def site_inputs(ids):
    xs = {n: [] for n in site_names()}
    for i in range(0, ids.shape[0], 4):
        for n in site_names(): T.site(n).cache_input = True
        T(ids[i:i + 4])
        for n in site_names():
            st = T.site(n); xs[n].append(st.last_input.reshape(-1, st.last_input.shape[-1])); st.cache_input = False
    return {n: torch.cat(v) for n, v in xs.items()}

X = site_inputs(torch.tensor(tok[0:16, :512].astype(np.int64), device=dev))
vpdlike = start in ('vpd', 'vpdgroup')
# Starts: svd, vpd, vpdgroup, neuron (the privileged axis: each MLP neuron's c_fc row and down_proj
# column one exact part under one gate on its own pre-activation).
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
    with full_float32():
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
    elif start == 'neuron':
        # The privileged axis: neuron j of the layer's MLP is one part, its c_fc row w_j (a c_fc slice
        # reading w_j . x and writing e_j) and its down_proj column (a down_proj slice reading e_j and
        # writing W_down[:, j]) under one gate (below); every part on is M exactly.
        V, U = (W.T.clone(), torch.eye(W.shape[0], device=dev)) if n.endswith('c_fc') else (torch.eye(W.shape[1], device=dev), W.T.clone())
    else:
        V, U = load(n + '.V'), load(n + '.U')
    if EXACT and start == 'neuron':
        raise SystemExit('the neuron start is exact without frames (DESCENT_FREEZE=1 keeps it exact)')
    if EXACT:
        # The start's reads kept (the canonical dual is an involution: the frame R (R^T R)^-1 has dual
        # R), its writes replaced by the exact ones. The SVD start's down_proj has fewer slices than
        # inputs, so its frame is the hidden (neuron) axis: f_i = e_i, slice i = neuron i.
        if (start == 'svd' or NEURON_DOWN) and n.endswith('down_proj'):
            F0 = torch.eye(W.shape[1], device=dev)
        else:
            Rd = V.T.cpu().double().numpy()
            F0 = torch.tensor(Rd @ np.linalg.inv(Rd.T @ Rd), dtype=torch.float32, device=dev)
        with torch.no_grad():
            V, U = frame(F0, W)
    with torch.no_grad():
        # A neuron's gate reads its own pre-activation w_j . x, signed (the neuron is on when it is
        # large and positive), threshold one per layer at the quantile matching VPD's mean count of
        # the layer's MLP slices, halved (a neuron part counts its two rank-one slices).
        r = (X[n] @ V) if start == 'neuron' else (X[n] @ V).abs() * U.norm(dim=1)
        s = 0.1 * r.pow(2).mean(0).sqrt().clamp_min(1e-12)
        if start == 'svd':
            tau = -3 * s
        elif start == 'neuron':
            l_ = n.split('.')[1]
            q = 1 - (VPD_COUNTS[f'h.{l_}.mlp.c_fc'] + VPD_COUNTS[f'h.{l_}.mlp.down_proj']) / 2 / V.shape[1]
            flat = r.reshape(-1)
            idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            tau = torch.full_like(s, torch.quantile(flat[idx], q).item())
        else:
            q = 1 - START_SCALE * VPD_COUNTS[n] / V.shape[1]
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
if start == 'neuron':
    # Neuron j's down_proj slice fires with its c_fc slice's gate; the part counts 2 when on.
    for l in range(4):
        fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
        GROUP[dn] = (fc, torch.arange(P[dn]['V'].shape[1], device=dev))
        MULT[fc] = torch.full((P[fc]['V'].shape[1],), 2.0, device=dev)
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

NH, HD = T.n_head, T.hd


def head_frame(F, W, o):
    """Reads V [H, d_in_h, C] and writes U [H, C, d_out_h] of the head blocks of W from frames F [H, C, HD]."""
    with full_float32():
        return head_frame_(F, W, o)


def head_frame_(F, W, o):
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


def head_output(c, U, o, swaps=()):
    """The map's output [tokens, d_out] from the gated coefficients c [H, tokens, C]; for o, `swaps`
    (h1, h2, token rows) replaces head h1's contribution by head h2's on those rows."""
    y = c @ U
    if not o:
        return y.permute(1, 0, 2).reshape(c.shape[1], -1)
    out = y.sum(0)
    for h1, h2, rows in swaps:
        out = out.index_add(0, rows, y[h2, rows] - y[h1, rows])
    return out


# DESCENT_ATTN_FREE=1: the heads' slices start at the exact frames' reads and writes and then train
# freely (no dual to recompute); with DESCENT_RESID each attention map's leftover keeps every part on
# equal to M.
ATTN_FREE = os.environ.get('DESCENT_ATTN_FREE') == '1'
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
        q = 1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / (NH * V.shape[-1])
        flat = r.reshape(-1); idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
        tau = torch.full_like(s_, torch.quantile(flat[idx], q).item())
        err = (head_output(c, U, o) - x @ W.T).abs().max().item()
        sv = torch.linalg.svdvals(F0.cpu().double())
        cond = (sv[:, 0] / sv[:, -1]).max().item()
    A[n] = {'F': F0.clone().requires_grad_(), 'V': V, 'U': U, 'tau': tau.clone().requires_grad_(), 's': s_, 'o': o}
    if ATTN_FREE:
        A[n]['V'], A[n]['U'] = V.clone().requires_grad_(), U.clone().requires_grad_()
    print(n, 'slices', NH * V.shape[-1], 'all-on error', err, 'worst head frame cond', round(cond, 1), flush=True)
SHARE_A = {}
# DESCENT_RESID=1 (MLP maps): a priced residual part per map. R = W^T - V U, recomputed from the current
# slices, is one more part, so every part on is M exactly by construction whatever the slices become.
# It is gated on its own read ||x R|| (the norm of its output on the map's own input): Phi((||x R|| -
# tau_R)/s_R) in training, hard at evaluation, with s_R = 0.1 x the root mean square of ||x W^T|| on
# M's fit tokens (the map's output scale; R starts at 0 for an exact start) and tau_R started at 3 s_R
# (off). On, it counts its rank-one equivalent, min(d_in, d_out) (a full-rank residual is that many
# rank-one units). Under DESCENT_F its entries are described like any part's (below), so a large
# residual costs bits and budget and a small one is cheap: F decides how far the slices drift.
RESID = os.environ.get('DESCENT_RESID') == '1'
# Typical input and output norms of every map on M's fit tokens (random weight edits are sized by them).
with torch.no_grad():
    TYPICAL = {n: (X[n].norm(dim=-1).pow(2).mean().sqrt().item(), (X[n] @ T.site(n).W.T).norm(dim=-1).pow(2).mean().sqrt().item())
               for n in X}
RES = {}
def assembled(n):
    """Map n's slices summed into one matrix [d_in, d_out] (an attention map's heads in place)."""
    if n in P:
        return P[n]['V'] @ P[n]['U']
    p = A[n]; blocks = p['V'] @ p['U']                                           # [H, d_in_h, d_out_h]
    return blocks.reshape(-1, blocks.shape[-1]) if p['o'] else blocks.permute(1, 0, 2).reshape(blocks.shape[1], -1)
if RESID:
    with torch.no_grad():
        for n in mlp + (attn if ATTN_FREE else []):
            W = T.site(n).W
            s_R = 0.1 * (X[n] @ W.T).norm(dim=-1).pow(2).mean().sqrt()
            RES[n] = {'tau': (3 * s_R).clone().requires_grad_(), 's': s_R, 'rank': float(min(W.shape)),
                      'ls': torch.full((W.shape[1], W.shape[0]), math.log(1e-3 * W.abs().mean().item()), device=dev).requires_grad_()}
def residual(n, x):
    """Map n's residual part on its input x: R = W^T - V U (or its posterior sample), gated on ||x R||."""
    R = RES[n]['sample'] if RES[n].get('sample') is not None else T.site(n).W.T - assembled(n)
    out = x @ R
    for b, kind, G, a in state['entry'].get(n, ()):
        idx = torch.tensor([b], device=out.device)
        if kind == 'noop':
            continue
        if kind == 'swap':
            c1, c2 = (torch.arange(h * HD, (h + 1) * HD, device=out.device) for h in G)
            out = out.index_add(0, idx, (x[b][..., c2] @ R[c2] - x[b][..., c1] @ R[c1])[None])
        elif kind == 'in':
            out = out.index_add(0, idx, (a * (x[b][..., G] @ R[G]))[None])
        else:
            out = out.index_add(0, idx, (a * (x[b] @ R[:, G]) @ torch.eye(out.shape[-1], device=out.device)[G])[None])
    if state['drop_R']:
        out = out.index_fill(0, torch.tensor(state['drop_R'], device=out.device), 0.0)
    if state['mode'] == 'all':
        return out * 0.0 if state.get('drop_leftover') else out
    z = (out.norm(dim=-1) - RES[n]['tau']) / RES[n]['s']
    hard = (z > 0).float()
    state['hard'].append(hard.reshape(-1) * RES[n]['rank'])
    state['resid_on'].setdefault(n, []).append(hard.mean().item())
    if state['mode'] == 'hard':
        return out * hard[..., None]
    phi = 0.5 * (1 + torch.erf(z / SQ2))
    state['soft'].append(phi.reshape(-1) * RES[n]['rank'])
    return out * phi[..., None]

del X

state = {'mode': 'M', 'soft': [], 'hard': [], 'gates': {}, 'route': {}, 'share': {}, 'r': {}, 'writers': {}, 'on': {}, 'gate': {}, 'edges_soft': [], 'edges_hard': [], 'share_a': {}, 'resid_on': {}, 'wedits_M': {}, 'wedits_P': {}, 'entry': {}, 'force_on': [], 'drop_R': []}
SQ2 = math.sqrt(2)
def make(n):
    st = T.site(n); p = P[n]; grp = GROUP.get(n); mult = MULT.get(n)
    layer = int(n.split('.')[1]); router = ROUTER.get(layer)
    if grp is not None:
        tied, own = grp[1] >= 0, grp[1].clamp_min(0)
    cur = {}
    def emit(coef):
        # A down_proj slice's coefficient on each token (its read times its gate): what it writes.
        if n.endswith('down_proj'):
            state['writers'][layer] = coef
        y = coef @ p['U']
        for b, kind, G, a in state['entry'].get(n, ()):
            if kind == 'out':
                # Output coordinates G of every part's write scaled by 1 + a.
                y = y.index_add(0, torch.tensor([b], device=y.device), (a * (coef[b] @ p['U'][:, G]) @ torch.eye(y.shape[-1], device=y.device)[G])[None])
        return y + residual(n, cur['x']) if RESID else y
    def fwd(x):
        cur['x'] = x
        if state['mode'] == 'M':
            return x @ st.W.T
        c = x @ p['V']
        un = un0 = p['U'].norm(dim=1)
        for b, kind, G, a in state['entry'].get(n, ()):
            if kind == 'in':
                # Input coordinates G of every part's read scaled by 1 + a.
                c = c.index_add(0, torch.tensor([b], device=c.device), (a * (x[b][..., G] @ p['V'][G]))[None])
            else:
                # Every part's write norm with output coordinates G scaled (its own read is |c| ||u||).
                un_b = (un0.pow(2) + ((1 + a) ** 2 - 1) * p['U'][:, G].pow(2).sum(1)).clamp_min(0).sqrt()
                un = un.expand(c.shape[0], 1, -1).clone() if un.dim() == 1 else un
                un[b] = un_b
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
        read = x @ p['G'] if ARM == 'dir' else c if start == 'neuron' else c.abs() * un
        if state.get('calib') is not None:
            state['calib'].setdefault(n, []).append(read.detach().reshape(-1))
        if router is not None:
            if n.endswith('c_fc'):
                state['route'][layer] = torch.relu(x @ router['G1'])
            read = state['route'][layer] @ router[n] if ARM == 'router_pure' else read + state['route'][layer] @ router[n]
        z = (read - p['tau']) / p['s']
        hard = (z > 0).float()
        phi = 0.5 * (1 + torch.erf(z / SQ2))
        if state['force_on']:
            on = torch.tensor(state['force_on'], device=z.device)
            hard = hard.index_fill(0, on, 1.0); phi = phi.index_fill(0, on, 1.0)
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


def share_o(l, c, p):
    """(hard, soft) gates [H, tokens, C] of layer l's o slices: each fires with a v part of its head
    (from this pass's post-attention parts) or alone on its own read of the head's output."""
    S = SHARE_A[l]
    hp, pp = state['share_a'][l]
    Aw = torch.softmax(S['L_o'], -1)                                           # [H, Co, 8]
    z = (c.abs() * p['U'].norm(dim=-1)[:, None, :] - p['tau'][:, None, :]) / p['s'][:, None, :]
    ho, po = (z > 0).float(), 0.5 * (1 + torch.erf(z / SQ2))
    idx = S['cand_o'].clamp_min(0)
    choice = Aw.argmax(-1)                                                     # [H, Co]
    part = idx.gather(-1, choice[..., None]).squeeze(-1)
    hard = torch.where((choice == 0)[:, None, :], ho, hp.gather(-1, part[:, None, :].expand(-1, hp.shape[1], -1)))
    soft = Aw[:, None, :, 0] * po + sum(Aw[:, None, :, k] * pp.gather(-1, idx[:, None, :, k].expand(-1, pp.shape[1], -1)) for k in range(1, 8))
    return hard, soft

def swaps_of(n, T_):
    """Head swaps on map n (o_proj) in the installed edits: (h1, h2, the edited sequence's token rows)."""
    return [(h[0], h[1], torch.arange(b * T_, (b + 1) * T_, device=dev)) for b, kind, h, _ in state['entry'].get(n, ()) if kind == 'swap']

def force_rows(hard, soft, T_):
    """The all-on sequences' gates [H, B*T, C] forced on (their token rows b*T .. (b+1)*T - 1)."""
    if not state['force_on']:
        return hard, soft
    rows = torch.cat([torch.arange(b * T_, (b + 1) * T_, device=hard.device) for b in state['force_on']])
    return hard.index_fill(1, rows, 1.0), soft.index_fill(1, rows, 1.0)

def make_attn(n):
    st = T.site(n); p = A[n]
    def fwd(x):
        if state['mode'] == 'M':
            return x @ st.W.T
        xin = x.reshape(-1, x.shape[-1])
        c = head_coefficients(xin, p['V'], p['o'])
        if p['o'] and state['entry'].get(n):
            # A head edit scales the head's input coordinates of o_proj: every o slice of that head reads
            # them, so its coefficient on that sequence is scaled by 1 + a (its gate reads the scaled value).
            scale = torch.ones(NH, c.shape[1], 1, device=c.device)
            for b, kind, G, a in state['entry'][n]:
                if kind == 'in':
                    scale[int(G[0]) // HD, b * x.shape[1]:(b + 1) * x.shape[1]] = 1 + a
            c = c * scale
        if state.get('capture') is not None and p['o']:
            state['capture'].setdefault(n, []).append(c.abs() * p['U'].norm(dim=-1)[:, None, :])
        if SHARE_A and p['o'] and state['mode'] != 'all':
            hard, soft = force_rows(*share_o(int(n.split('.')[1]), c, p), x.shape[1])
            state['hard'].append(hard.sum((0, 2)))
            if state['mode'] == 'hard':
                g = hard
            else:
                state['soft'].append(soft.sum((0, 2)))
                g = soft
            y = head_output(c * g, p['U'], p['o'], swaps_of(n, x.shape[1])).view(x.shape[0], x.shape[1], -1)
            return y + residual(n, x) if n in RES else y
        if state['mode'] == 'all':
            g = 1.0
        else:
            if state.get('calib') is not None:
                state['calib'].setdefault(n, []).append((c.abs() * p['U'].norm(dim=-1)[:, None, :]).reshape(-1))
            z = (c.abs() * p['U'].norm(dim=-1)[:, None, :] - p['tau'][:, None, :]) / p['s'][:, None, :]
            hard, phi = force_rows((z > 0).float(), 0.5 * (1 + torch.erf(z / SQ2)), x.shape[1])
            state['hard'].append(hard.sum((0, 2)))
            if state['mode'] == 'hard':
                g = hard
            else:
                state['soft'].append(phi.sum((0, 2)))
                g = phi if gate == 'mf' else hard + phi - phi.detach()
        y = head_output(c * g, p['U'], p['o'], swaps_of(n, x.shape[1])).view(x.shape[0], x.shape[1], -1)
        return y + residual(n, x) if n in RES else y
    return fwd
for n in attn: T.site(n)._forward = make_attn(n)

# Weight edits (DESCENT_WEDITS = edited sequences per training batch): an edit is an additive
# Delta W = A B^T on one or more of M's maps, defined in M's terms and applied verbatim to both models
# at every use of the map: M's output W x and P's (its parts plus leftover, which together own W) both
# gain Delta W x on the edited sequence (install(): M every edit additively, P neuron-group edits entry-wise).
def with_edits(n, f):
    def g(x):
        y = f(x)
        for b, A_, B_ in state['wedits_M' if state['mode'] == 'M' else 'wedits_P'].get(n, ()):
            y = y.index_add(0, torch.tensor([b], device=x.device), ((x[b] @ B_) @ A_.T)[None])
        return y
    return g
plain_forward = {n: T.site(n)._forward for n in site_names()}
for n in site_names():
    T.site(n)._forward = with_edits(n, plain_forward[n])

KINDS_ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_fc', 'down_proj')
FAMILIES = ('neurons_remove', 'neurons_scale', 'neurons_in_remove', 'neurons_in_scale', 'head_remove', 'head_scale', 'head_swap', 'random')
def site_of(l, k):
    return f"h.{l}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}"

def draw_edit(g, family):
    """One weight edit of M, as {map: (A, B)}, from the generator g. Neuron edits act on a group of a
    layer's neurons, its size log-uniform from 1 to all 3,072 (one neuron alone moves M by about 2e-4
    bits): on their down_proj columns (neurons_*) or their c_fc rows (neurons_in_*), removed (zeroed) or
    scaled by 0.5, 2 or 3; head edits on the head's o_proj
    columns, a head swap puts head h2's q, k, v rows and o columns in head h1's place, and a random edit
    is A B^T on one map with rank log-uniform from 1 to full and size log-uniform from 5% to 100% of
    the map's typical output norm."""
    l = int(g.integers(4)); W = lambda k: T.site(site_of(l, k)).W
    eye = lambda d, idx: torch.eye(d, device=dev)[:, idx]
    if family.startswith('neurons'):
        width = W('down_proj').shape[1]
        size = max(1, int(round(math.exp(g.uniform(0, math.log(width))))))
        G = sorted(int(j) for j in g.choice(width, size, replace=False))
        a = -1.0 if family.endswith('remove') else float(g.choice([0.5, 2.0, 3.0])) - 1
        Gt = torch.tensor(G, device=dev)
        if family.startswith('neurons_in'):
            return {site_of(l, 'c_fc'): (eye(width, G), a * W('c_fc')[G].T), '_entry': {site_of(l, 'c_fc'): ('out', Gt, a)}}
        return {site_of(l, 'down_proj'): (a * W('down_proj')[:, G], eye(width, G)), '_entry': {site_of(l, 'down_proj'): ('in', Gt, a)}}
    nh, hd = T.n_head, T.hd
    cols = lambda h: list(range(h * hd, (h + 1) * hd))
    if family in ('head_remove', 'head_scale'):
        h = int(g.integers(nh)); a = -1.0 if family == 'head_remove' else float(g.choice([0.5, 2.0, 3.0])) - 1
        return {site_of(l, 'o_proj'): (a * W('o_proj')[:, cols(h)], eye(W('o_proj').shape[1], cols(h))),
                '_entry': {site_of(l, 'o_proj'): ('in', torch.tensor(cols(h), device=dev), a)}}
    if family == 'head_swap':
        h1, h2 = (int(x) for x in g.choice(nh, 2, replace=False))
        out = {site_of(l, k): (eye(W(k).shape[0], cols(h1)), (W(k)[cols(h2)] - W(k)[cols(h1)]).T) for k in ('q_proj', 'k_proj', 'v_proj')}
        out[site_of(l, 'o_proj')] = (W('o_proj')[:, cols(h2)] - W('o_proj')[:, cols(h1)], eye(W('o_proj').shape[1], cols(h1)))
        # Entry-wise on P: head h1's q, k, v, o entries become head h2's inside every part and the
        # leftover, so head h1 computes what head h2 computes and its contribution to the stream is head
        # h2's; applied where that contribution is summed (o_proj), q, k, v left as they are.
        out['_entry'] = {site_of(l, k): ('noop', None, 0.0) for k in ('q_proj', 'k_proj', 'v_proj')}
        out['_entry'][site_of(l, 'o_proj')] = ('swap', (h1, h2), 0.0)
        return out
    k = KINDS_[int(g.integers(6))]; n = site_of(l, k); d_out, d_in = W(k).shape
    r = max(1, int(round(math.exp(g.uniform(0, math.log(min(d_out, d_in)))))))
    f = math.exp(g.uniform(math.log(0.05), 0.0))
    xs, ys = TYPICAL[n]
    c = f * ys / math.sqrt(d_out * r) / xs
    return {n: (torch.tensor(g.standard_normal((d_out, r)), dtype=torch.float32, device=dev) * math.sqrt(c),
                torch.tensor(g.standard_normal((d_in, r)), dtype=torch.float32, device=dev) * math.sqrt(c))}

def install(edits):
    """edits: per sequence of the batch, None, 'allon' (every part's gate on, the leftover off, against M),
    'lrm' (the leftover removed from both: M becomes its parts all on, P runs without the leftover), or
    {map: (A, B), '_entry': {map: (kind, G, a)}}: a weight edit, additive Delta W = A B^T on M, and on P
    entry-wise where '_entry' gives it (kind 'in': input coordinates G of the map scaled by 1 + a, in
    every part's read and the leftover; 'out': output coordinates G, in every part's write and the
    leftover), additive otherwise."""
    state['wedits_M'], state['wedits_P'], state['entry'] = {}, {}, {}
    state['force_on'], state['drop_R'] = [], []
    for b, e in enumerate(edits):
        if e == 'allon':
            state['force_on'].append(b); state['drop_R'].append(b); continue
        if e == 'lrm':
            state['drop_R'].append(b); continue
        entry = (e or {}).get('_entry', {})
        for n, v in (e or {}).items():
            if n == '_entry':
                continue
            state['wedits_M'].setdefault(n, []).append((b, *v))
            if n in entry and (n in P or n in A):
                state['entry'].setdefault(n, []).append((b, *entry[n]))
            else:
                state['wedits_P'].setdefault(n, []).append((b, *v))

N_EDITS = int(os.environ.get('DESCENT_WEDITS', '0'))
# DESCENT_ALLON=1: one sequence of every batch runs with every part's gate on and the leftover off,
# against M (ties the parts' sum to M). DESCENT_LRM=1 (with DESCENT_RESID): one sequence runs the
# leftover-removal experiment, Delta W = -R on both models (M becomes its parts all on; P runs without
# its leftover), tying P's sparse run to its own dense run.
ALLON = int(os.environ.get('DESCENT_ALLON', '0') == '1')
LRM = int(os.environ.get('DESCENT_LRM', '0') == '1' and RESID)
# Held-out weight edits: a fixed set, 4 per held-out sequence (32 in all), families in turn, seed 7.
_g = np.random.default_rng(7)
EVAL_EDITS = [(i, FAMILIES[(4 * i + k) % len(FAMILIES)]) for i in range(ev.shape[0]) for k in range(4)]
EVAL_EDITS = [(i, fam, draw_edit(_g, fam)) for i, fam in EVAL_EDITS]

@torch.no_grad()
def evaluate_edits():
    """Held-out weight-edit gaps, KL(M_e || P_e) in bits per token (P hard), per family and per bin of
    the edit's effect on M, KL(M_e || M): below 0.01, 0.01-0.1, 0.1-1, above 1 bits."""
    rows = {}
    for i in range(ev.shape[0]):
        install([None]); rows[i] = run(ev[i:i + 1], 'M')
    by_fam, by_bin = {}, {}
    for j in range(0, len(EVAL_EDITS), 4):
        chunk = EVAL_EDITS[j:j + 4]
        ids = torch.cat([ev[i:i + 1] for i, _, _ in chunk])
        install([e for _, _, e in chunk])
        lm_e = run(ids, 'M'); lp_e = run(ids, 'hard')
        gap = kl_bits(lm_e, lp_e).mean(-1); eff = kl_bits(lm_e, torch.cat([rows[i] for i, _, _ in chunk])).mean(-1)
        for (_, fam, _), g_, f_ in zip(chunk, gap.tolist(), eff.tolist()):
            by_fam.setdefault(fam, []).append((g_, f_))
            b_ = '<0.01' if f_ < 0.01 else '0.01-0.1' if f_ < 0.1 else '0.1-1' if f_ < 1 else '>1'
            by_bin.setdefault(b_, []).append(g_)
    install([None])
    return {'gap_mean': float(np.mean([g_ for v in by_fam.values() for g_, _ in v])),
            'by_family': {k: {'gap': round(float(np.mean([g_ for g_, _ in v])), 4), 'effect': round(float(np.mean([f_ for _, f_ in v])), 4), 'n': len(v)} for k, v in by_fam.items()},
            'by_effect': {k: {'gap': round(float(np.mean(v)), 4), 'n': len(v)} for k, v in by_bin.items()}}

# Attention gates after attention: a v slice's coefficient c_i(s) on each key position s is mixed by
# the head's attention pattern into m_i(t) = sum_s a(t, s) c_i(s) at the query position t, and the
# slice is gated there, on its own post-attention read r_i(t) = |m_i(t)| ||f_i|| (what it delivers to
# the head's output at t; causal). Its contribution to the head's output is f_i m_i(t) g_i(t), so with
# every gate on the head's output is M's exactly. q and k slices keep their own reads at their own
# positions (they shape the pattern); o slices read the head's output, which is already post-attention.
# With DESCENT_ARM=share the head's v slices form parts gated by the norm of their members'
# post-attention reads, and o slices join a v part of their head or keep their own read.
def attn_v(l, h, pattern):
    n = f'h.{l}.attn.v_proj'; p = A[n]
    B_, T_ = h.shape[0], h.shape[1]
    c = head_coefficients(h.reshape(-1, h.shape[-1]), p['V'], False)           # [H, B*T, C]
    m = pattern @ c.view(NH, B_, T_, -1).permute(1, 0, 2, 3)                   # [B, H, T, C]
    if state['mode'] == 'all':
        g = 1.0
        if state.get('capture') is not None:
            state['capture'].setdefault(n, []).append((m.abs() * p['U'].norm(dim=-1)[None, :, None, :]).permute(1, 0, 2, 3).reshape(NH, -1, m.shape[-1]))
    else:
        r = m.abs() * p['U'].norm(dim=-1)[None, :, None, :]
        if SHARE_A:
            S = SHARE_A[l]
            rr = r.permute(1, 0, 2, 3).reshape(NH, -1, r.shape[-1])               # [H, tokens, C]
            Aw = torch.softmax(S['L_v'], -1)
            R2 = torch.zeros_like(rr)
            for k in range(8):
                R2 = R2.scatter_add(-1, S['cand_v'][:, None, :, k].expand_as(rr), rr.pow(2) * Aw[:, None, :, k])
            zp = ((R2 + 1e-12).sqrt() - S['t'][:, None, :]) / S['s'][:, None, :]
            hp, pp = (zp > 0).float(), 0.5 * (1 + torch.erf(zp / SQ2))
            state['share_a'][l] = (hp, pp)
            pick = S['cand_v'].gather(-1, Aw.argmax(-1, keepdim=True)).squeeze(-1)
            hard = hp.gather(-1, pick[:, None, :].expand_as(hp))
            soft = sum(Aw[:, None, :, k] * pp.gather(-1, S['cand_v'][:, None, :, k].expand_as(pp)) for k in range(8))
            back = lambda t: t.view(NH, B_, T_, -1).permute(1, 0, 2, 3)
            hard, soft = back(hard), back(soft)
        else:
            if state.get('calib') is not None:
                state['calib'].setdefault(n, []).append(r.detach().reshape(-1))
            z = (r - p['tau'][None, :, None, :]) / p['s'][None, :, None, :]
            hard = (z > 0).float(); soft = 0.5 * (1 + torch.erf(z / SQ2))
        if state['force_on']:
            # The all-on sequences' gates on.
            on = torch.tensor(state['force_on'], device=hard.device)
            hard, soft = hard.index_fill(0, on, 1.0), soft.index_fill(0, on, 1.0)
        state['hard'].append(hard.sum((1, 3)).reshape(-1))
        if state['mode'] == 'hard':
            g = hard
        else:
            state['soft'].append(soft.sum((1, 3)).reshape(-1))
            g = soft if gate == 'mf' or SHARE_A else hard + soft - soft.detach()
    y = (m * g) @ p['U'][None]                                                 # [B, H, T, HD]
    if n in RES:
        # v_proj's leftover, gated on its own read at the key position, mixed by the pattern like any value.
        dv = residual(n, h).view(B_, T_, NH, HD).transpose(1, 2)                 # [B, H, T, HD]
        y = y + pattern @ dv
    for b, Ae, Be in state['wedits_P'].get(n, ()):
        # A weight edit of v_proj adds Delta W x to the values, which the pattern mixes like any value.
        dv = ((h[b] @ Be) @ Ae.T).view(T_, NH, HD).transpose(0, 1)               # [H, T, HD]
        y = y.index_add(0, torch.tensor([b], device=y.device), (pattern[b] @ dv)[None])
    return y.transpose(1, 2).reshape(B_, T_, -1)

plain_hidden = T.hidden
def hidden(ids):
    """M's forward (mode M) or P's: vpd_model's Target.hidden with the attention pattern explicit, so
    that v slices are gated after it."""
    if state['mode'] == 'M':
        return plain_hidden(ids)
    B_, T_ = ids.shape
    x = T.wte[ids]
    causal = torch.ones(T_, T_, dtype=torch.bool, device=x.device).tril()
    for i in range(T.n_layer):
        site = lambda k: T.site(f"h.{i}.{'mlp' if k in ('c_fc', 'down_proj') else 'attn'}.{k}")
        h = vpd_model.rms(x, T.norms[2 * i], T.eps)
        q = T._rope(site('q_proj')(h).view(B_, T_, NH, HD).transpose(1, 2), T_)
        k = T._rope(site('k_proj')(h).view(B_, T_, NH, HD).transpose(1, 2), T_)
        pattern = ((q @ k.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(~causal, float('-inf')).softmax(-1)
        x = x + site('o_proj')(attn_v(i, h, pattern))
        h = vpd_model.rms(x, T.norms[2 * i + 1], T.eps)
        x = x + site('down_proj')(vpd_model.gelu_tanh(site('c_fc')(h)))
    return vpd_model.rms(x, T.ln_f, T.eps)
if attn:
    T.hidden = hidden
    # Calibration on P's own run with every slice on (= M): each v slice's noise scale and the v map's
    # threshold from the post-attention reads (the quantile matching VPD's mean count at the map).
    with torch.no_grad():
        state['capture'] = {}
        for i in range(0, 4, 2):
            state['mode'], state['soft'], state['hard'] = 'all', [], []
            T(torch.tensor(tok[i:i + 2, :512].astype(np.int64), device=dev))
        cap = {n: torch.cat(v, 1) for n, v in state['capture'].items()}
        state['capture'] = None
        for l in range(4):
            n = f'h.{l}.attn.v_proj'; r = cap[n]
            A[n]['s'] = 0.1 * r.pow(2).mean(1).sqrt().clamp_min(1e-12)
            q = 1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / r.shape[0] / r.shape[2]
            flat = r.reshape(-1); idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            A[n]['tau'] = torch.full_like(A[n]['s'], torch.quantile(flat[idx], q).item()).requires_grad_()
        if ARM == 'share':
            for l in range(4):
                v, o = f'h.{l}.attn.v_proj', f'h.{l}.attn.o_proj'
                gv = (cap[v] > A[v]['tau'][:, None, :]).float(); go = (cap[o] > A[o]['tau'][:, None, :]).float()
                nv, no = gv.sum(1), go.sum(1)
                vv = nv[:, :, None] + nv[:, None, :] - 2 * gv.transpose(1, 2) @ gv
                vv.diagonal(dim1=1, dim2=2).fill_(-1.0)
                cand_v = vv.topk(8, dim=2, largest=False).indices                  # [H, Cv, 8], own first
                vo = nv[:, :, None] + no[:, None, :] - 2 * gv.transpose(1, 2) @ go
                near = vo.topk(7, dim=1, largest=False).indices.transpose(1, 2)    # [H, Co, 7]
                cand_o = torch.cat([torch.full_like(near[..., :1], -1), near], -1)
                L_v = torch.zeros(NH, cand_v.shape[1], 8, device=dev); L_v[..., 0] = 6.0
                L_o = torch.zeros(NH, cand_o.shape[1], 8, device=dev); L_o[..., 0] = 6.0
                SHARE_A[l] = {'cand_v': cand_v, 'cand_o': cand_o, 'L_v': L_v.requires_grad_(), 'L_o': L_o.requires_grad_(),
                              't': A[v]['tau'].detach().clone().requires_grad_(), 's': A[v]['s'].clone()}
        del cap

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
            if not (NEURON_DOWN and n.endswith('down_proj')):
                P[n]['V'], P[n]['U'] = frame(P[n]['F'], T.site(n).W)
    for n in attn if not ATTN_FREE else ():
        A[n]['V'], A[n]['U'] = head_frame(A[n]['F'], T.site(n).W, A[n]['o'])
    state['collect'] = {} if SHARE else None
    state['resid_on'] = {}
    install([None])
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
        if RES:
            # M with its leftover removed: every part on, the leftover off (= W - R in every map).
            state['drop_leftover'] = True
            lp = run(ids, 'all'); r.setdefault('kl_parts_on', []).append(kl_bits(lm, lp).mean().item())
            state['drop_leftover'] = False
    out = {k: float(np.mean(v)) for k, v in r.items() if k != 'per_map' and v}
    out['per_map'] = [round(float(x), 2) for x in np.mean(r['per_map'], 0)]
    if RES:
        out['residual'] = {n: {'norm_rel_W': round(((T.site(n).W.T - assembled(n)).norm() / T.site(n).W.norm()).item(), 4),
                               'fires': round(float(np.mean(state['resid_on'].get(n, [0.0]))), 4)} for n in RES}
        if FMODE:
            out['residual_bits'] = residual_bits()
    state['resid_on'] = {}
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
    if SHARE_A:
        # Attention parts under the hardened assignment, per layer: ranks (v members plus o slices
        # that joined) over the heads' parts that have a v member.
        out['attn_parts'] = []
        for l, S in SHARE_A.items():
            pick = S['cand_v'].gather(-1, S['L_v'].argmax(-1, keepdim=True)).squeeze(-1)    # [H, Cv]
            ch = S['L_o'].argmax(-1)
            po = torch.where(ch == 0, torch.full_like(ch, -1), S['cand_o'].gather(-1, ch[..., None]).squeeze(-1))
            rk = torch.zeros(NH, pick.shape[1], device=dev).scatter_add_(-1, pick, torch.ones_like(pick, dtype=torch.float))
            has = rk > 0
            rk = rk + torch.zeros_like(rk).scatter_add_(-1, po.clamp_min(0), (po >= 0).float())
            rk = rk[has]
            out['attn_parts'].append({'parts': int(has.sum()), 'o_on_own_read': int((po < 0).sum()), 'mean_rank': round(rk.mean().item(), 2),
                                      'max_rank': int(rk.max().item()), 'rank_1': int((rk == 1).sum()), 'rank_2_7': int(((rk >= 2) & (rk <= 7)).sum()),
                                      'rank_8+': int((rk >= 8).sum())})
    state['collect'] = None
    return out

# The VPD start's thresholds set in the gated run: each map's threshold is the quantile of its reads that
# matches its target count (VPD's per-map count scaled to the budget) with every upstream map gated at its
# own threshold, one pass per map in forward order (after pass j the first j maps are at their fixed
# point). Thresholds set on M's inputs left maps dead whose reads shrink under upstream gating (the
# whole model at K = 128: layer 0's v and o and layer 1's o at 0.0-0.1 on against 1.2-3.2 targeted, and a
# gate far below its threshold gets no gradient back).
if start == 'vpd' and not (SHARE or SHARE_A or ROUTER or EXACT or ARM == 'dir'):
    target = {n: 1 - START_SCALE * VPD_COUNTS[n] / P[n]['V'].shape[1] for n in mlp}
    target.update({n: 1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / (NH * A[n]['V'].shape[-1]) for n in attn})
    ids_c = torch.tensor(tok[0:4, :512].astype(np.int64), device=dev)
    with torch.no_grad():
        install([None])
        for _ in range(len(target)):
            state['calib'] = {}
            run(ids_c, 'hard')
            for n, v in state['calib'].items():
                flat = torch.cat(v); idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
                (P[n] if n in P else A[n])['tau'].fill_(torch.quantile(flat[idx].float(), target[n]).item())
        state['calib'] = None
        run(ids_c, 'hard')
        print('thresholds set in the gated run: parts on per token', round(torch.stack(state['hard']).sum(0).mean().item(), 1),
              [round(h.mean().item(), 2) for h in state['hard']], flush=True)

# Every trained tensor as (container, key, scale): Adam steps of 0.3% of the scale per step, the
# scale a weight tensor's root mean square (a router G2 started at zero: its map's noise scale) and a
# threshold's its map's noise scale x 33. DESCENT_LR multiplies every step size (default 1).
LR = float(os.environ.get('DESCENT_LR', '1'))
rms = lambda q: q.detach().pow(2).mean().sqrt().item()
# DESCENT_FREEZE=1: the slices stay as they start (M's own, for the neuron start) and only the gates
# train, so every part on is M exactly at every step.
FREEZE = os.environ.get('DESCENT_FREEZE') == '1'
slots = [(P[n], w, rms(P[n][w])) for n in mlp for w in (('F',) if EXACT else ('V', 'U')) + (('G',) if ARM == 'dir' else ())
         if not (EXACT and NEURON_DOWN and n.endswith('down_proj') and w == 'F') and not (FREEZE and w in ('V', 'U', 'F'))]
slots += [(A[n], w, rms(A[n][w])) for n in attn for w in (('V', 'U') if ATTN_FREE else ('F',))] + [(A[n], 'tau', 100 / 3 * A[n]['s'].mean().item()) for n in attn]
for l, S in SHARE_A.items():
    slots += [(S, 't', 100 / 3 * S['s'].mean().item()), (S, 'L_v', 20 / 3), (S, 'L_o', 20 / 3)]
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
# DESCENT_COUNT=mean (F only): the budget's E[k] at the posterior mean, the explanation evaluated, rather
# than at the step's sample. At the sample, parts whose reads or thresholds are wide fire at random (the whole
# model at K = 128 under F: 602 parts on per training token against 77 at the mean after 915 steps, then the
# multiplier rose to 914 and closed every gate at the mean).
COUNT_MEAN = FMODE and os.environ.get('DESCENT_COUNT') == 'mean'
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
groups += [{'params': [Rn['tau']], 'lr': LR * 0.1 * Rn['s'].item()} for Rn in RES.values()]
if FMODE:
    groups += [{'params': [Rn['ls']], 'lr': LR * 1e-2} for Rn in RES.values()]
opt = torch.optim.Adam(groups)
trainable = [q for g in groups for q in g['params']]

def draw(mean):
    """Install the posterior mean (mean) or one sample of every tensor (F only), then (exact) every
    map's reads and writes from its frame."""
    for cont, key, mu, ls in leaves:
        cont[key] = mu if mean else mu + ls.exp() * torch.randn_like(mu)
    for n, Rn in RES.items():
        mu_R = T.site(n).W.T - assembled(n)
        Rn['mean'] = mu_R
        Rn['sample'] = mu_R if mean or not FMODE else mu_R + Rn['ls'].exp() * torch.randn_like(mu_R)
    if EXACT:
        for n in mlp:
            if not (NEURON_DOWN and n.endswith('down_proj')):
                P[n]['V'], P[n]['U'] = frame(P[n]['F'], T.site(n).W)
    for n in attn if not ATTN_FREE else ():
        A[n]['V'], A[n]['U'] = head_frame(A[n]['F'], T.site(n).W, A[n]['o'])

# DESCENT_PRIOR=part: each part's read and write get their own prior group N(0, v_j) (v_j at its optimum,
# the mean of mu^2 + sigma^2 over the part's entries; each v_j described in (1/2) ln n_j nats, n_j its
# entries) instead of one per tensor. Under a tensor's prior an unused part's entries revert to the
# tensor's scale, the scale of the used parts, and such random reads fire at the sampled parameters (the
# whole model at K = 128: sigma 2x RMS(mu) on reads and writes after 915 steps, 602 parts on per training
# token against 77 at the posterior mean); under its own prior an unused part costs nothing at mu = 0
# whatever its width, and its mean decays to zero.
PART_PRIOR = os.environ.get('DESCENT_PRIOR') == 'part'

def description_bits():
    """KL(q || p) in bits (F only)."""
    total = 0.0
    for _, key, mu, ls in leaves:
        e = mu.pow(2) + (2 * ls).exp()
        if PART_PRIOR and key in ('V', 'U') and mu.dim() >= 2:
            # A read's part is its last axis (MLP [d_in, r], heads [H, d, C]), a write's its second last
            # (MLP [r, d_out], heads [H, C, d_out]).
            d_ = -2 if key == 'V' else -1
            v = e.mean(d_)
            total = total + 0.5 * (mu.shape[d_] * torch.log(v).sum() - 2 * ls.sum()) + 0.5 * math.log(mu.shape[d_]) * v.numel()
            continue
        v = e.mean()
        total = total + 0.5 * (mu.numel() * torch.log(v) - 2 * ls.sum())
    for Rn in RES.values():
        # The residual's entries under their own prior group N(0, v_R), posterior N(R, sigma_R^2).
        v = (Rn['mean'].pow(2) + (2 * Rn['ls']).exp()).mean()
        total = total + 0.5 * (Rn['mean'].numel() * torch.log(v) - 2 * Rn['ls'].sum())
    return total / math.log(2)

def residual_bits():
    """The residual parts' share of the description, in bits (F only)."""
    total = 0.0
    for Rn in RES.values():
        v = (Rn['mean'].pow(2) + (2 * Rn['ls']).exp()).mean()
        total += 0.5 * (Rn['mean'].numel() * torch.log(v) - 2 * Rn['ls'].sum()).item()
    return total / math.log(2)
# The budget's multiplier (library_mdl's rule, Settings::budget): the step descends KL + lam (E[k] - K),
# then lam <- max(0, lam + eta (E[k] - K)) with eta = lam_hat / (K B), lam_hat = |<g_F, g_k>| / |g_k|^2
# the multiplier at which the budget's gradient cancels the KL gradient's component along
# g_k = dE[k]/dtheta (both measured on this step), B the batches in one pass over the training rows;
# lam starts at the first step's lam_hat (the balance value), so the budget term begins at the strength
# that holds E[k] where it is.
B = train_rows * 512 / (batch * seq)
DUAL = os.environ.get('DESCENT_DUAL', 'measured')
# DESCENT_DUAL=logint: the budget term lambda (E[k] - K) with log lambda integrated, log lambda <- log
# lambda + (E[k] - K) / (K B_H), lambda started at the median over thresholds of the balance
# |dF/dtau| / |dE[k]/dtau| (edits' rule in the Rust fitter); B_H, the horizon in steps, is a tenth of the
# evaluation interval (about 0.5M tokens), as a whole pass over the training rows is never reached here.
B_H = max(1, EVAL // 10)
lam, rng = 0.0, np.random.default_rng(0)
log = {'start': start, 'K': K, 'steps': steps, 'gate': gate, 'arm': ARM, 'dual': DUAL, 'train_rows': train_rows, 'F': FMODE, 'edges': EDGES, 'trace': []}
# DESCENT_SAVE=PATH: after every evaluation, the maps' slices and gates at the posterior mean
# (reads V [d_in, C], writes U [C, d_out], thresholds tau [C] and noise scales s [C] per map, and the
# neuron start's tied down slices), for export_to_rust.py (library_vpd's importer).
def save(step):
    if os.environ.get('DESCENT_SAVE'):
        torch.save({'step': step, 'start': start, 'arm': ARM, 'gate': gate,
                    'maps': {n: {k: P[n][k].detach().float().cpu() for k in ('V', 'U', 'tau', 's')} for n in mlp},
                    # The heads' slices (DESCENT_SITES=all): reads V [H, d_in_h, C], writes U [H, C, d_out_h].
                    'attn': {n: {k: A[n][k].detach().float().cpu() for k in ('V', 'U', 'tau', 's')} for n in attn},
                    'tied': {dn: (fc, own.cpu()) for dn, (fc, own) in GROUP.items()}}, os.environ['DESCENT_SAVE'])

draw(True)
e = evaluate(); e['weight_edits'] = evaluate_edits(); print('start', e, flush=True); log['trace'].append({'step': 0, **e}); save(0)
g_edits = np.random.default_rng(11)
step_seconds = []
t0 = time.time()
for step in range(steps):
    rows = rng.integers(0, train_rows, batch); rows = np.where(rows >= 1024, rows + 8, rows); offs = rng.integers(0, 513 - seq, batch)
    ids = torch.tensor(np.stack([tok[r, o:o + seq] for r, o in zip(rows, offs)]).astype(np.int64), device=dev)
    t_step = time.time()
    kinds = [draw_edit(g_edits, FAMILIES[(step * N_EDITS + b) % len(FAMILIES)]) if b < N_EDITS else None for b in range(batch)]
    if ALLON:
        kinds[N_EDITS] = 'allon'
    if LRM:
        kinds[N_EDITS + ALLON] = 'lrm'
    install(kinds)
    with torch.no_grad():
        lm = run(ids, 'M')
    draw(False)
    if LRM:
        # The leftover-removal experiment: M with Delta W = -R is the parts all on; its target is computed
        # at this step's parts (no gradient), and P runs the same sequence without its leftover.
        b_ = N_EDITS + ALLON
        with torch.no_grad():
            install([None]); state['drop_leftover'] = True
            lm = lm.index_copy(0, torch.tensor([b_], device=dev), run(ids[b_:b_ + 1], 'all'))
            state['drop_leftover'] = False
        install(kinds)
    lp = run(ids, 'soft')
    kl_seq = kl_bits(lm, lp).mean(-1)
    kl = kl_seq.mean()
    # The objective's data and description terms (F; without DESCENT_F only the data term).
    desc = description_bits() if FMODE else torch.zeros((), device=dev)
    # Each kept edge's index bits, at the expected gates.
    edge_bits = sum(E['bits'] * (0.5 * (1 + torch.erf(E['eta'] / SQ2))).sum() for E in EDGE.values()) if EDGES else torch.zeros((), device=dev)
    objective = kl + (desc + edge_bits) / N
    # The per-token budget is the explanation's execution on clean text: counted over the clean
    # sequences only (an edited, all-on or leftover-removal sequence runs extra machinery, e.g. its
    # leftover where the parts cannot express the edit, which the clean budget does not cap).
    counted = torch.tensor([1.0 if k_ is None else 0.0 for k_ in kinds], device=dev)[:, None].expand(batch, seq).reshape(-1)
    if COUNT_MEAN:
        # Under F the budget counts the explanation evaluated, the posterior mean: the clean sequences run
        # again at the mean (the data term stays at this step's sample).
        clean = [b for b, k_ in enumerate(kinds) if k_ is None]
        draw(True); install([None] * len(clean))
        run(ids[clean], 'soft')
        counted = torch.ones(len(clean) * seq, device=dev)
    ek = (torch.stack(state['soft']).sum(0) * counted).sum() / counted.sum()
    hk = ((torch.stack(state['hard']).sum(0) * counted).sum() / counted.sum()).item()
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
    if DUAL == 'logint':
        if step == 0:
            # lambda starts at the median over thresholds of the balance |dF/dtau| / |dE[k]/dtau|.
            taus = [mu for _, key, mu, _ in leaves if key == 'tau'] if FMODE else [cont[key] for cont, key, _ in slots if key == 'tau']
            gF = torch.autograd.grad(objective, taus, retain_graph=True, allow_unused=True)
            gk = torch.autograd.grad(ek, taus, retain_graph=True, allow_unused=True)
            ratio = torch.cat([(a.abs() / b.abs()).reshape(-1)[b.abs().reshape(-1) > 0] for a, b in zip(gF, gk) if a is not None and b is not None])
            lam = max(ratio.median().item(), 1e-12) if ratio.numel() else 1e-3
        opt.zero_grad(); (objective + lam * (ek - K)).backward(); opt.step()
        # The relative violation, capped at +1 as it is bounded by -1 below, so lambda rises no faster than
        # it can fall (an uncapped rise ran away at K = 64 while the count started at 3.7 K).
        lam = lam * math.exp(min((ek.item() - K) / K, 1.0) / B_H)
        g_f = None
    elif DUAL == 'fixed':
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
    if dev == 'cuda':
        torch.cuda.synchronize()
    step_seconds.append(time.time() - t_step)
    last = step == steps - 1 or time.time() - t0 > LIMIT
    if (step + 1) % EVAL == 0 or last:
        # This step's parts on per clean training token, per map (at the sample under F), and under F each
        # kind of tensor's posterior width: the median over tensors of RMS(sigma) / RMS(mu).
        train_per_map = [((h * counted).sum() / counted.sum()).item() for h in state['hard']]
        sigma_rel = {}
        for _, key, mu, ls in leaves:
            sigma_rel.setdefault(key, []).append((ls.exp().pow(2).mean() / mu.pow(2).mean().clamp_min(1e-30)).sqrt().item())
        sigma_rel = {key: float(np.median(v)) for key, v in sigma_rel.items()}
        draw(True)
        e = evaluate(final=last)
        e['weight_edits'] = evaluate_edits()
        e['step_seconds'] = float(np.mean(step_seconds)) if step_seconds else None
        e['train_kl_edited'] = float(kl_seq[:N_EDITS].mean()) if N_EDITS else None
        step_seconds = []
        rec = {'step': step + 1, 'lambda': lam, 'K_t': Kt, 'train_kl': kl.item(), 'description_bits': desc.item(), 'train_edge_bits': float(edge_bits),
               'train_F': objective.item(), 'train_k_soft': ek.item(), 'train_k_hard': hk, 'train_per_map': train_per_map,
               'sigma_rel': sigma_rel, **e,
               'seconds': time.time() - t0}
        log['trace'].append(rec); print(rec, flush=True); save(step + 1)
        json.dump(log, open(out, 'w'), indent=1)
    if last:
        break

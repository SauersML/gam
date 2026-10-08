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
lambda >= 0 by the dual step (DESCENT_DUAL, below) until E[k] <= K.
Starts:
  svd  the input-whitened SVD of each map (W C^(1/2) = U S V^T on M's inputs of rows 0..15;
       reads C^(-1/2) v, writes s u; all on = M), thresholds 3 s_i below zero, so every gate is on
  vpd  VPD's MLP slices (s-55ea3f9b), one threshold per map set so the active count on the fit
       tokens matches VPD's causal-importance count at that map (fitmath's KL 3.28 start), then
       trained per slice
  vpdgroup  as vpd, with each down_proj slice tied to the gate of the c_fc slice of its layer whose
       firing is nearest its own (below)
Gate arms (DESCENT_ARM): own (above); dir, a separate signed gate direction per slice; share, learned
gate sharing across the slices of a layer (below); rot (below).
Objective (DESCENT_F=1): F, the bits-back code length per training token (below); otherwise KL alone.
Exactness (DESCENT_EXACT=1): each map's slices are parametrized by a read frame F and its canonical
dual, so they sum to M's weight at every step whatever F becomes (frame(), below); training moves
only the frames and the gates, and kl_all_on stays at rounding.
Wiring (DESCENT_EDGES=1, own arm): the MLP parts' reads of the residual stream are an explicit, fitted
graph (below), each kept edge charged its index bits.
Dual step (DESCENT_DUAL): logint (default; log lambda integrated toward E[k] = K, started at the balance rate)
or pin (lambda held at the balance rate, and after every step one common threshold shift puts the delivered
hard program's bits per token at K).
Main line (DESCENT_ARM=rot, below): exact learned blocks (rotated groups of slices of each map, one gate per
block), the per-token budget in description bits, the attention blocks across heads.
Training rows 0..1023 of tokens.f64, held-out evaluation rows 1024..1031 (4096 tokens), where VPD's
causal-importance masks give KL 0.737 at 129 active MLP slices per token.
Usage: budget_descent.py START K STEPS OUT.json GATE EVAL SECONDS GAM TARGET VPD TOKENS
  GATE     mf = expected gate forward and derivative; st = hard gate forward, expected gate derivative;
           bern = (rot) a hard gate drawn on with probability Phi(z), Phi(z)'s derivative
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
# DESCENT_CONCEPTS (default 1; the rot arm): the objective is the concepts a reader follows per clean token, those of
# the blocks on (1 for the gate plus, per slice, its read, its write and its scales: 4 for an MLP or OV slice, whose
# neuron-space or head-space direction inside the block is not counted, 3 for a q or k slice) plus the gates
# evaluated (every non-empty block's, the head scores read by score gates, the mean parts, a gate network's
# parameters), minimized subject to the delivered (hard) program's KL on clean text at most kappa (DESCENT_KAPPA,
# default 0.553 bits per token, VPD's published whole-model KL): faithfulness is a requirement, not a price. The
# multiplier mu follows the step's delivered KL on its clean sequences (fresh text, so unseen) by log-integral
# ascent. The start has every block on (thresholds at the 1% quantile of each block set's reads), where KL is about
# 0. Bits-back (DESCENT_F) is off. Blocks never on in an evaluation's hard passes are pruned (off for good; all-on
# keeps them, so P stays exact). (Levin's per-token KL + log2 concepts, tried first, traded faithfulness for
# concepts at one bit per halving: hard KL rose from 3.1 to 6.2 as the count fell.)
# On the slice arms (DESCENT_CONCEPTS=1 there) a slice is 4 concepts (gate, read, write, scale), its gate one of the
# gates evaluated, and a map's or head's mean part one concept.
CONCEPTS = os.environ.get('DESCENT_CONCEPTS', '1' if ARM == 'rot' else '0') == '1'
KAPPA = float(os.environ.get('DESCENT_KAPPA', '0.553'))
CPS = 1 if ARM == 'rot' else 4                                                  # concepts per unit on (rot counts concepts)
start_q = lambda q: 0.01 if CONCEPTS else q                                      # the start's threshold quantile: all on
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
# The attention maps gated as slices (all of them, except under the rot arm, whose attention blocks are below).
sliced = [] if os.environ.get('DESCENT_ARM') == 'rot' else attn
tok = np.memmap(TOKENS, dtype=np.uint16 if TOKENS.endswith('.u16') else np.float64, mode='r').reshape(-1, 513)
ev = torch.tensor(tok[1024:1032, :512].astype(np.int64), device=dev)
# Training rows: DESCENT_TRAIN_ROWS rows of the file, skipping the held-out rows 1024..1031 (default 1024).
# DESCENT_BATCH sequences of DESCENT_SEQ tokens per step (default 32 x 512, the largest that fits an A40 on the whole
# model: 33 GB, 16,179 tokens/s against 14,911 at 16 x 512 and 8,660 at 8 x 256, compiled).
train_rows = int(os.environ.get('DESCENT_TRAIN_ROWS', '1024'))
batch, seq = int(os.environ.get('DESCENT_BATCH', '32')), int(os.environ.get('DESCENT_SEQ', '512'))
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
# VPD's slices are read for the VPD starts' MLP maps and for the heads (DESCENT_SITES=all, any start).
if (vpdlike or attn or ARM == 'rot') and str(VPD_DIR).endswith('.pth'):
    raw = torch.load(str(VPD_DIR), map_location='cpu', weights_only=True, mmap=True)
    load = lambda k: raw['_components.' + k.rsplit('.', 1)[0].replace('.', '-') + '.' + k.rsplit('.', 1)[1]].float().to(dev)
elif vpdlike or attn or ARM == 'rot':
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
        raise SystemExit('the neuron start is exact without frames')
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
            q = start_q(1 - START_SCALE * (VPD_COUNTS[f'h.{l_}.mlp.c_fc'] + VPD_COUNTS[f'h.{l_}.mlp.down_proj']) / 2 / V.shape[1])
            flat = r.reshape(-1)
            idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            tau = torch.full_like(s, torch.quantile(flat[idx], q).item())
        else:
            q = start_q(1 - START_SCALE * VPD_COUNTS[n] / V.shape[1])
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
    for l in range(T.n_layer):
        fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
        GROUP[dn] = (fc, torch.arange(P[dn]['V'].shape[1], device=dev))
        MULT[fc] = torch.full((P[fc]['V'].shape[1],), 2.0, device=dev)
if start == 'vpdgroup':
    with torch.no_grad():
        for l in range(T.n_layer):
            fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
            on = lambda m: (((X[m] @ P[m]['V']).abs() * P[m]['U'].norm(dim=1)) > P[m]['tau']).float()
            gf, gd = on(fc), on(dn)
            mism = gf.sum(0)[:, None] + gd.sum(0)[None, :] - 2 * (gf.T @ gd)
            bm, best = mism.min(0)
            owner = torch.where(bm < gd.sum(0), best, torch.full_like(best, -1))
            GROUP[dn] = (fc, owner)
            MULT[fc] = 1 + torch.bincount(owner[owner >= 0], minlength=gf.shape[1]).float()
            print(dn, 'tied', int((owner >= 0).sum()), 'of', owner.numel(), 'to', int((MULT[fc] > 1).sum()), 'c_fc slices', flush=True)
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
        for l in range(T.n_layer):
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
    for l in range(1, T.n_layer):
        n_w = sum(P[f'h.{k}.mlp.down_proj']['V'].shape[1] for k in range(l))
        EDGE[l] = {'eta': torch.full((n_w, P[f'h.{l}.mlp.c_fc']['V'].shape[1]), 3.0, device=dev, requires_grad=True), 'bits': math.log2(n_w)}
        print(f'layer {l}: {n_w} x {EDGE[l]["eta"].shape[1]} candidate edges, {EDGE[l]["bits"]:.2f} bits each', flush=True)
    # The pre-MLP norm's scale r of P's own full stream, recorded per layer as the model computes it.
    MLP_NORM = {id(T.norms[2 * l + 1]): l for l in range(T.n_layer)}
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
# freely (no dual to recompute).
ATTN_FREE = os.environ.get('DESCENT_ATTN_FREE') == '1'
A = {}
for n in sliced:
    if start not in ('vpd', 'neuron'):
        raise SystemExit('DESCENT_SITES=all: the vpd or neuron start (the heads from VPD\'s attention slices)')
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
        q = start_q(1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / (NH * V.shape[-1]))
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
# Typical input and output norms of every map on M's fit tokens (random weight edits are sized by them).
with torch.no_grad():
    TYPICAL = {n: (X[n].norm(dim=-1).pow(2).mean().sqrt().item(), (X[n] @ T.site(n).W.T).norm(dim=-1).pow(2).mean().sqrt().item())
               for n in X}

# The rot arm's neuron grouping reads M's c_fc inputs.
X_FC = {n: X[n] for n in mlp if n.endswith('c_fc')} if ARM == 'rot' else {}
# Every map's mean input on M's fit tokens (the slice arms' mean parts: a slice's mean coefficient is its read of it).
XBAR = {n: X[n].mean(0) for n in X}
del X

state = {'mode': 'M', 'soft': [], 'hard': [], 'gates': {}, 'share': {}, 'r': {}, 'writers': {}, 'on': {}, 'gate': {}, 'edges_soft': [], 'edges_hard': [], 'share_a': {}, 'wedits_M': {}, 'wedits_P': {}, 'entry': {}, 'force_on': [], 'gn_in': {}, 'score': {}, 'mlp_x': {}}
SQ2 = math.sqrt(2)
# DESCENT_SCOREGATE=1 (slice arms): every attention slice's gate also reads its head's largest attention logit at the
# slice's token, s_h(t) = max_{u <= t} q_t . k_u / sqrt(hd) from M's q and k maps on P's own stream (standardized per
# head on the first pass), z += beta s_h(t) with one beta per slice (started at 0, so the start is the own read). A
# read of the residual stream cannot see a repeat; the head's own scores can (toys' induction toy: AUC 0.91-0.97 at
# the second copy's tokens, own reads 0.33-0.75).
SCOREGATE = os.environ.get('DESCENT_SCOREGATE') == '1'
SCORE_NORM = {}
# DESCENT_READSIDE=1 (slice arms): a down_proj slice's gate reads its direction on the layer's pre-activation,
# |v . (W_fc x)| ||u|| with x the normed stream entering the MLP (the read side, in the stream), in place of its own
# read of the activation, |v . GELU(h)| ||u|| (toys' superposition toy: stream-side gates exact, neuron-space gates
# at Jaccard 0.51). A c_fc slice's own read is already its direction on the stream.
READSIDE = os.environ.get('DESCENT_READSIDE') == '1'
# DESCENT_MEANQK=1 (rot attention): each head's q and k get a positional part, their mean (pre-RoPE, on the start's
# calibration text), always on; the q and k blocks split q - mean and k - mean. With every q and k block off a head
# attends by position alone (RoPE of the means), not uniformly: toys found L1H1 (previous token) keeps 97% of its
# effect in that form and L1H2/H4/H5 nearly content-free. Each costs one concept (its write) per head and map,
# counted with the gates evaluated, as it runs on every token.
MEANQK = os.environ.get('DESCENT_MEANQK') == '1'
# DESCENT_MEANPARTS=1 (rot): off means typical, not zero. Each MLP gets always-on mean parts, its mean pre-activation
# in c_fc's output and its mean activation in down_proj's input (so W_dn E[act] is written on every token), the MLP
# blocks splitting p - E[p] and act - E[act]; each head's OV gets its mean value, the OV blocks splitting the mixed
# value less it (the pattern's rows sum to 1, so pattern (v - E[v]) = mixed value - E[v]). A block off is then that
# block mean-ablated, not zeroed (GELU's mean is not zero), and every part on is still M. A mean part is one
# concept (its write) per map or head, as are MEANQK's (counted with the gates evaluated).
MEANPARTS = os.environ.get('DESCENT_MEANPARTS') == '1'
# DESCENT_DESTKV=1 (slice arms): a k slice is gated at the reading token (the query t), not where it sits: on at t
# when its largest share of t's scores, max over u <= t of |c(u) (q_t . rope_u(u_c))| / sqrt(hd) (c(u) its
# coefficient at the key u, u_c its write, q_t M's query from P's stream), passes its threshold; off, its term leaves
# t's scores only. (v slices are gated at the query already: on the pattern-mixed coefficients.) A query then reads
# only the k slices on at it. Without it a query reads every k slice on anywhere in its prefix, and the concepts per
# token count that union (the honest count; toys: causal VPD reads 271 concepts per token on layer 0 so counted,
# against 23 per position).
DESTKV = os.environ.get('DESCENT_DESTKV') == '1'

def slice_mean(n):
    """Whether map n's slices (slice arms) split their coefficient's deviation from its mean, the mean always on:
    q and k under MEANQK, every other map under MEANPARTS. A slice off is then mean-ablated, not zeroed."""
    qk = n.endswith(('q_proj', 'k_proj'))
    return (MEANQK and qk) or (MEANPARTS and not qk)
# DESCENT_GATE_H=1 (rot): each gate's binary entropy H(Phi(z)) at each token is charged in bits per token: the bits
# to specify a draw the gate does not determine. Training on sampled gates (bern) then drives the gates to 0 or 1,
# so the delivered program (on where z > 0) is the trained one.
GATE_H = os.environ.get('DESCENT_GATE_H') == '1'

def slice_gates(read, c, un, tau, s, taun, ex):
    """A slice map's gates (hard, expected) from its own reads (read, or |c| un when None) and thresholds (tau, s and
    the negative side's taun or None, broadcast against c; ex: the gate network's term, or None)."""
    if taun is None:
        z = ((c.abs() * un if read is None else read) - tau) / s
    else:
        z = torch.maximum(c * un - tau, -c * un - taun) / s
    if ex is not None:
        z = z + ex
    return (z > 0).float(), 0.5 * (1 + torch.erf(z / SQ2))

_slice_compiled = []
def slice_gates_run(*a):
    """slice_gates(*a), compiled (torch.compile; inductor fuses the chain and its backward) on CUDA outside
    calibration (speed: the gate ladder's slice arms ran these element-wise passes one kernel each)."""
    if dev != 'cuda' or state.get('calib') is not None:
        return slice_gates(*a)
    if not _slice_compiled:
        _slice_compiled.append(torch.compile(slice_gates))
    return _slice_compiled[0](*a)

def slice_apply(read, c, un, tau, s, taun, ex, red, mode):
    """slice_gates' gates applied: (c times the hard gate in a hard pass, else times the training gate; the hard
    gates' per-token sum over red; the expected gates' sum, or None in a hard pass; the hard gates in a hard pass)."""
    hard, phi = slice_gates(read, c, un, tau, s, taun, ex)
    if mode == 'hard':
        return c * hard, hard.sum(red), None, hard
    return c * (phi if gate == 'mf' else hard + phi - phi.detach()), hard.sum(red), phi.sum(red), None

_apply_compiled = []
def slice_apply_run(*a):
    """slice_apply(*a), compiled on CUDA outside calibration (the gated coefficients, their sums and the backward
    in the gates' own fused pass)."""
    if dev != 'cuda' or state.get('calib') is not None:
        return slice_apply(*a)
    if not _apply_compiled:
        _apply_compiled.append(torch.compile(slice_apply))
    return _apply_compiled[0](*a)

def make(n):
    st = T.site(n); p = P[n]; grp = GROUP.get(n); mult = MULT.get(n)
    layer = int(n.split('.')[1])
    if grp is not None:
        tied, own = grp[1] >= 0, grp[1].clamp_min(0)
    cur = {}
    def emit(coef):
        # A down_proj slice's coefficient on each token (its read times its gate): what it writes.
        if cur.get('cm') is not None:
            coef = coef + cur['cm']                                             # the mean part
        if n.endswith('down_proj'):
            state['writers'][layer] = coef
        y = coef @ p['U']
        for b, kind, G, a in state['entry'].get(n, ()):
            if kind == 'out':
                # Output coordinates G of every part's write scaled by 1 + a.
                y = y.index_add(0, torch.tensor([b], device=y.device), (a * (coef[b] @ p['U'][:, G]) @ torch.eye(y.shape[-1], device=y.device)[G])[None])
        return y
    def fwd(x):
        cur['x'], cur['cm'] = x, None
        if state.get('dense') is not None and n.endswith('c_fc'):
            state['dense'][(layer, 1)] = x
        if state['mode'] == 'M':
            return x @ st.W.T
        if READSIDE and n.endswith('c_fc'):
            state['mlp_x'][layer] = x
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
        if slice_mean(n):
            cur['cm'] = XBAR[n] @ p['V']
            c = c - cur['cm']
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
        read = x @ p['G'] if ARM == 'dir' else c if start == 'neuron' else None
        if READSIDE and n.endswith('down_proj'):
            read = ((state['mlp_x'][layer] @ T.site(f'h.{layer}.mlp.c_fc').W.T) @ p['V']).abs() * un
        if state.get('calib') is not None:
            state['calib'].setdefault(n, []).append((c.abs() * un if read is None else read).detach().reshape(-1))
        ex = None
        if n in GN:
            # The layer's gate network on the MLP's own input (the normed stream entering c_fc).
            if n.endswith('c_fc'):
                state['gn_in'][('mlp', layer)] = x
            ex = gate_net_s(n, state['gn_in'][('mlp', layer)])
        if grp is None and mult is None and not EDGES and not state['force_on']:
            coef, hb, sb, hard = slice_apply_run(read, c, un, p['tau'], p['s'], p.get('taun'), ex, (-1,), state['mode'])
            if hard is not None:
                state['on'][n] = hard
            state['hard'].append(hb.reshape(-1))
            if state['mode'] == 'hard':
                state['on'][n] = hard
            else:
                state['soft'].append(sb.reshape(-1))
            return emit(coef)
        hard, phi = slice_gates_run(read, c, un, p['tau'], p['s'], p.get('taun'), ex)
        if state['force_on']:
            on = torch.tensor(state['force_on'], device=c.device)
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
        if state.get('dense') is not None and n.endswith('q_proj'):
            state['dense'][(int(n.split('.')[1]), 0)] = x
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
            return head_output(c * g, p['U'], p['o'], swaps_of(n, x.shape[1])).view(x.shape[0], x.shape[1], -1)
        cm = None
        if state['mode'] == 'all':
            g = 1.0
        else:
            if slice_mean(n):
                cm = head_coefficients(XBAR[n][None], p['V'], p['o'])                 # [H, 1, C], the mean part
                c = c - cm
            if state.get('calib') is not None:
                state['calib'].setdefault(n, []).append((c.abs() * p['U'].norm(dim=-1)[:, None, :]).reshape(-1))
            ex = None
            if n in GN:
                # The layer's gate network on the stream entering the attention (q's input; o reads it too).
                if not p['o']:
                    state['gn_in'][('attn', n.split('.')[1])] = x
                g_ = gate_net_s(n, state['gn_in'][('attn', n.split('.')[1])])
                ex = g_.reshape(-1, NH, c.shape[-1]).permute(1, 0, 2)
            if 'beta' in p:
                # The head's largest attention logit at the slice's token (the query's for q and o, the key's for k).
                sg = state['score'][int(n.split('.')[1])].permute(1, 0, 2).reshape(NH, -1, 1) * p['beta'][:, None, :]
                ex = sg if ex is None else ex + sg
            args = (None, c, p['U'].norm(dim=-1)[:, None, :], p['tau'][:, None, :], p['s'][:, None, :], p['taun'][:, None, :] if 'taun' in p else None, ex)
            if CONCEPTS and n.endswith('k_proj'):
                # k slices gated at their keys: a query reads every one on anywhere in its prefix, so each token counts
                # that union (straight-through: the hard union's value, the expected union's gradient).
                hard, phi = force_rows(*slice_gates_run(*args), x.shape[1])
                B_, T_, C_ = x.shape[0], x.shape[1], c.shape[-1]
                hu = hard.view(NH, B_, T_, C_).cummax(2).values
                state['hard'].append(hu.sum((0, 3)).reshape(-1))
                if state['mode'] == 'hard':
                    g = hard
                    state['on'][n] = hard
                else:
                    pu = 1 - torch.cumsum(torch.log1p(-phi.clamp_max(1 - 1e-6)).view(NH, B_, T_, C_), 2).exp()
                    state['soft'].append((pu if gate == 'mf' else hu + pu - pu.detach()).sum((0, 3)).reshape(-1))
                    g = phi if gate == 'mf' else hard + phi - phi.detach()
                return head_output(c * g if cm is None else c * g + cm, p['U'], p['o'], swaps_of(n, x.shape[1])).view(x.shape[0], x.shape[1], -1)
            if not state['force_on']:
                cg, hb, sb, hd = slice_apply_run(*args, (0, 2), state['mode'])
                state['hard'].append(hb)
                if sb is not None:
                    state['soft'].append(sb)
                if hd is not None:
                    state['on'][n] = hd
                return head_output(cg if cm is None else cg + cm, p['U'], p['o'], swaps_of(n, x.shape[1])).view(x.shape[0], x.shape[1], -1)
            hard, phi = force_rows(*slice_gates_run(*args), x.shape[1])
            state['hard'].append(hard.sum((0, 2)))
            if state['mode'] == 'hard':
                g = hard
            else:
                state['soft'].append(phi.sum((0, 2)))
                g = phi if gate == 'mf' else hard + phi - phi.detach()
        return head_output(c * g if cm is None else c * g + cm, p['U'], p['o'], swaps_of(n, x.shape[1])).view(x.shape[0], x.shape[1], -1)
    return fwd
for n in sliced: T.site(n)._forward = make_attn(n)


# DESCENT_ARM=rot (neuron start, MLP maps, F): parts are learned blocks of each MLP layer's neuron space, exact
# by construction. A layer's neurons fall into groups of DESCENT_ROT (default 32) co-firing neurons (greedy on the
# correlations of M's activations); a group's orthonormal basis Q = exp(S), S skew and learned (started at 0, so
# Q = I), splits it into rotated slices: slice i is the direction Q_i of the group's neuron space, its c_fc part
# reading W_fc,G^T Q_i and writing Q_i, its down_proj part reading Q_i of the activation and writing W_dn,G Q_i.
# A group's slices fall into blocks under a learned assignment (started one slice per block), and a block's one
# gate switches its slices in both maps: a block is a subspace of the group's neuron space, of any rank, that
# passes before and after the activation when on (h_G = Q diag(g) Q^T p_G, z_G = Q diag(g) Q^T a_G). With every
# block on, Q Q^T = I and P is M exactly, whatever Q.
# A block's gate reads its own write, z_j = (sqrt(sum_{i in j} r_i^2) - tau_j) / s_j with r_i = |abar_G . Q_i|
# ||W_dn,G Q_i||, abar the layer's activation with all of the layer on (computed from P's own input).
# Cost: slice i's description L_i in bits (bits-back): its dense read (c_fc, d_in numbers) and write (down, d_out
# numbers) at the slice's own posterior widths under a zero-mean prior per layer and map, plus half of each
# rotation angle it shares; block j's cost L_j is its slices', its threshold's and log2 of the number of blocks
# (its index). The per-token budget is the expected bits of the blocks on, E_t[sum_on L_j] <= B, B the bits of
# K of VPD's MLP subcomponents under the same rule (VPD's per-map counts times K / 129, each subcomponent at a
# width of 1% of its tensor's root mean square, as ours start, plus its index).
ROT, ROTG = {}, int(os.environ.get('DESCENT_ROT', '32'))
# DESCENT_ROT_ATTN (default: a head's coordinates): the attention groups' size, in coordinates of the map's whole
# output (q, k, v) or input (o) across heads; 768 is the whole map, so a block's direction may span heads (as
# VPD's q and k subcomponents do: with groups inside a head the budget's one to four q slices per layer reached
# one to four heads, and the rest attended uniformly).
ROTGA = int(os.environ.get('DESCENT_ROT_ATTN', '0'))
# DESCENT_ATTN_START=vpd (with DESCENT_ROT_ATTN=768): the attention bases start at VPD's subcomponents' directions
# (below) instead of the maps' coordinates.
ATTN_START = os.environ.get('DESCENT_ATTN_START', 'coord')
# A slice's assignment logits start at ln(99 (g - 1)) on its own block and 0 elsewhere: 99% of its weight on its own
# block whatever the group size (a fixed 6 gives 93% at g = 32 and 34% at g = 768).
ASSIGN0 = lambda g: math.log(99 * (g - 1))
_masks = {}
def mask_of(g):
    """The strict upper triangle of a g x g matrix (where the angles live)."""
    if g not in _masks:
        _masks[g] = torch.triu(torch.ones(g, g, device=dev), 1)
    return _masks[g]
# A layer's neuron permutation and its inverse are gathers whose backward is the other permutation's gather
# (indexing's backward, an accumulating scatter through a sort, took 55 of 248 ms of a whole-model step's device
# time on an A40).
class _Permuted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, inv):
        ctx.save_for_backward(inv)
        return x.index_select(-1, idx)
    @staticmethod
    def backward(ctx, g):
        (inv,) = ctx.saved_tensors
        return g.index_select(-1, inv), None, None
def permuted(x, idx, inv):
    """x[..., idx] for a permutation idx with inverse inv."""
    return _Permuted.apply(x, idx, inv)
if ARM == 'rot':
    if start != 'neuron' or (os.environ.get('DESCENT_F') == '1') == CONCEPTS:
        raise SystemExit('DESCENT_ARM=rot: the neuron start, and F (DESCENT_F=1) or the concepts objective (DESCENT_CONCEPTS=1)')
    with torch.no_grad():
        for l in range(T.n_layer):
            fc, dn = f'h.{l}.mlp.c_fc', f'h.{l}.mlp.down_proj'
            Wf, Wd = T.site(fc).W, T.site(dn).W                                   # [3072, 768], [768, 3072]
            a_ = vpd_model.gelu_tanh(X_FC[fc] @ Wf.T)
            Z = (a_ - a_.mean(0)) / a_.std(0).clamp_min(1e-6)
            C = (Z.T @ Z / Z.shape[0]).cpu().numpy()
            free = np.ones(C.shape[0], bool); perm = []
            for j0 in np.argsort(-a_.abs().mean(0).cpu().numpy()):
                if not free[j0]:
                    continue
                cand = np.where(free)[0]
                pick = cand[np.argsort(-C[j0, cand])][:ROTG] if len(cand) > ROTG else cand
                pick = [j0] + [int(j) for j in pick if j != j0][:ROTG - 1]
                free[pick] = False; perm += pick
            perm = torch.tensor(perm, device=dev); ng = perm.numel() // ROTG
            Gf = Wf[perm].view(ng, ROTG, -1); Gd = Wd[:, perm].T.reshape(ng, ROTG, -1)
            ROT[l] = {'perm': perm, 'inv': torch.argsort(perm), 'ng': ng, 'g': ROTG, 'di': Wf.shape[1], 'do': Wd.shape[0],
                      'Gfc': Gf @ Gf.transpose(1, 2), 'Gdn': Gd @ Gd.transpose(1, 2),                 # [ng, g, g] Grams
                      'A': torch.zeros(ng, ROTG, ROTG, device=dev, requires_grad=True),
                      'L': (ASSIGN0(ROTG) * torch.eye(ROTG, device=dev)).repeat(ng, 1, 1).requires_grad_(),
                      'tau': torch.zeros(ng, ROTG, device=dev, requires_grad=True), 's': torch.ones(ng, ROTG, device=dev),
                      'ls_fc': torch.full((ng, ROTG), math.log(0.01 * Wf.pow(2).mean().sqrt().item()), device=dev, requires_grad=True),
                      'ls_dn': torch.full((ng, ROTG), math.log(0.01 * Wd.pow(2).mean().sqrt().item()), device=dev, requires_grad=True)}
    MASK = torch.triu(torch.ones(ROTG, ROTG, device=dev), 1)
ROT_ALL = list(ROT.values())


def rot_train_gate(hard, phi):
    """A block's gate in training. mf: the expected gate Phi(z); st: the hard gate (as the scorer runs it) with
    Phi(z)'s gradient; bern: on with probability Phi(z), drawn each pass (on or off, as the scorer runs it, and on
    average the expected gate), with Phi(z)'s gradient. Under mf the trained explanation scales blocks partially
    (the MLP's hard gates scored 1.83 bits per token at 5M against 1.14); the straight-through gate with a wider
    surrogate and the mean-field gate annealed to hard both lost (2.49 and 2.59 on the MLP)."""
    if gate == 'mf':
        return phi
    if gate == 'bern':
        return torch.bernoulli(phi.detach()) + phi - phi.detach()
    return hard + phi - phi.detach()

# The bases Q of every group set (the MLP's and attention's) are computed in one batched pass when the first is
# needed, and log2 of the number of blocks once per draw of the assignments, each kept while the same tensors are
# installed unmodified: one host wait per pass. Every other product of the rot arm runs in full float32 (qeinsum):
# under TF32's 10-bit mantissa the all-on pass drifted from M (9.0e-5 bits after 60 steps on the whole model,
# 2.0e-6 in float32 at the same step time).
_rot_memo = {}
def rot_memo(key, tensors, make):
    """make(), kept while the same tensors (by identity, unmodified since) are installed."""
    hit = _rot_memo.get(key)
    if hit is not None and len(hit[0]) == len(tensors) and all(a is b and v == b._version for a, v, b in zip(hit[0], hit[1], tensors)):
        return hit[2]
    out = make()
    _rot_memo[key] = (list(tensors), [t._version for t in tensors], out)
    return out

def qeinsum(eq, *ops):
    """torch.einsum in full float32 (a compiled core's caller holds float32 for all of it)."""
    if torch.compiler.is_compiling():
        return torch.einsum(eq, *ops)
    with full_float32():
        return torch.einsum(eq, *ops)

# The bases are computed in full float32 (TF32 off), the Taylor series after scaling S by its infinity norm (its
# largest row sum; the sum of all |S| entries took about 5 more squarings, each doubling the rounding error): on
# descent's trained angles the 32-groups are orthogonal to 1e-5 (2e-4 with the entry sum) and 768-group Cayley
# bases to 2e-6. Float64 ran at 1/64 of an A40's float32 rate (343 ms of a 1.5 s whole-model step); the float32
# drift that moved the bases to float64 had its products in TF32.
def rot_Q_all(As):
    """rot_Q of each angle tensor in As, the products batched (each group set squared as often as rot_Q
    squares it alone)."""
    with full_float32():
        return rot_Q_products(As)

def rot_Q_products(As):
    Ss = [A_ - A_.transpose(1, 2) for A_ in (a_ * mask_of(a_.shape[-1]) for a_ in As)]
    ks = [max(0, math.ceil(math.log2(max(v, 1e-12) / 0.25))) for v in torch.stack([S_.detach().abs().sum(-1).amax() for S_ in Ss]).tolist()]
    order = sorted(range(len(As)), key=lambda i: -ks[i])
    sizes = [As[i].shape[0] for i in order]
    X_ = torch.cat([Ss[i] / 2 ** ks[i] for i in order])
    E_ = torch.eye(X_.shape[-1], device=X_.device, dtype=X_.dtype).expand_as(X_) + X_; term = X_
    for j in range(2, 10):
        term = term @ X_ / j; E_ = E_ + term
    for m in range(ks[order[0]] if As else 0):
        n_ = sum(sz for sz, i in zip(sizes, order) if ks[i] > m)
        E_ = E_ @ E_ if n_ == E_.shape[0] else torch.cat((E_[:n_] @ E_[:n_], E_[n_:]))
    out = [None] * len(As)
    for part, i in zip(E_.split(sizes), order):
        out[i] = part.float()
    return out

def rot_cayley_all(Rs):
    """The Cayley bases (I + S)^-1 (I - S) of each (angles, start basis Q0 or None) in Rs, in one batched solve, in
    full float32 (a start basis's product under TF32 was orthogonal only to 1e-3)."""
    g = Rs[0][0].shape[-1]
    with full_float32():
        A_ = torch.cat([a * mask_of(g) for a, _ in Rs]); S_ = A_ - A_.transpose(1, 2)
        I_ = torch.eye(g, device=S_.device, dtype=S_.dtype).expand_as(S_)
        Q = torch.linalg.solve(I_ + S_, I_ - S_)
        return [Q0 @ q if Q0 is not None else q for q, (_, Q0) in zip(Q.split([a.shape[0] for a, _ in Rs]), Rs)]

def rot_Q(R):
    """The groups' bases Q = exp(S), S the skew part of A's strict upper triangle (Taylor series after
    scaling, then squaring: matrix products only, orthogonal to rounding), with every installed (or every
    posterior-mean) angle tensor's in one pass."""
    A = R['A']; g = A.shape[-1]
    if g > 64:
        # Large groups: the Cayley transform (I + S)^-1 (I - S), orthogonal for any skew S, every group set of this
        # size in one batched solve per pass.
        for name, Rs in (('installed', [(R_['A'], R_.get('Q0')) for R_ in ROT_ALL if R_['A'].shape[-1] == g]),
                         ('mean', [(R_['A_leaf'][0], R_.get('Q0')) for R_ in ROT_ALL if 'A_leaf' in R_ and R_['A'].shape[-1] == g])):
            at = [i for i, (t, _) in enumerate(Rs) if t is A]
            if at:
                return rot_memo((name, torch.is_grad_enabled(), g), [t for t, _ in Rs], lambda: rot_cayley_all(Rs))[at[0]]
        return rot_cayley_all([(A, R.get('Q0'))])[0]
    for name, As in (('installed', [R_['A'] for R_ in ROT_ALL if R_['A'].shape[-1] == g]),
                     ('mean', [R_['A_leaf'][0] for R_ in ROT_ALL if 'A_leaf' in R_ and R_['A'].shape[-1] == g])):
        at = [i for i, t in enumerate(As) if t is A]
        if at:
            Q = rot_memo((name, torch.is_grad_enabled(), g), As, lambda: rot_Q_all(As))[at[0]]
            break
    else:
        Q = rot_Q_all([A])[0]
    if R.get('Q0') is None:
        return Q
    with full_float32():
        return R['Q0'] @ Q

def rot_slice_bits(R, Q):
    """Per slice [ng, g]: its description in bits (dense read and write at its widths, its rotation angles' share)
    and its down write's norm ||W_dn,G Q_i||."""
    sf, sd = (2 * R['ls_fc']).exp(), (2 * R['ls_dn']).exp()
    nf = qeinsum('nki,nkm,nmi->ni', Q, R['Gfc'], Q); nd = qeinsum('nki,nkm,nmi->ni', Q, R['Gdn'], Q)
    di, do = R['di'], R['do']
    vf = (nf.sum() + di * sf.sum()) / (di * nf.numel()); vd = (nd.sum() + do * sd.sum()) / (do * nd.numel())
    if CONCEPTS:
        return torch.full_like(nd, 4.0), (nd.clamp_min(0) + 1e-30).sqrt()
    bits = 0.5 * (di * torch.log(vf / sf) + (nf + di * sf) / vf - di) + 0.5 * (do * torch.log(vd / sd) + (nd + do * sd) / vd - do)
    return bits / math.log(2) + rot_angle_bits(R), nd.clamp_min(0).sqrt()

def rot_angle_bits(R):
    """Per slice [ng, g]: half of each rotation angle it shares (F's posterior on the angles), in bits."""
    if 'A_leaf' not in R:
        return 0.0
    mu, ls = R['A_leaf']
    M_ = mask_of(mu.shape[-1])
    e = (mu.pow(2) + (2 * ls).exp()) * M_
    vA = e.sum() / (M_.sum() * mu.shape[0])
    kl = 0.5 * (torch.log(vA) - 2 * ls + e / vA - 1) * M_
    return 0.5 * (kl.sum(-1) + kl.sum(-2)) / math.log(2)

def rot_tau_bits(R):
    """Per block [ng, g]: its threshold's description in bits (the fitted-mean prior); under CONCEPTS its gate's concept."""
    if CONCEPTS:
        return torch.ones_like(R['s'])
    if 'tau_leaf' not in R:
        return torch.zeros_like(R['s'])
    mu, ls = R['tau_leaf']
    v = ((mu - mu.mean()).pow(2) + (2 * ls).exp()).mean()
    return 0.5 * (torch.log(v) - 2 * ls + ((mu - mu.mean()).pow(2) + (2 * ls).exp()) / v - 1) / math.log(2)

def rot_index_bits():
    """log2 of the number of blocks (MLP, OV and QK blocks with a slice or plane); under CONCEPTS the gate's concept
    (the start's per-block cost beside its slices')."""
    if CONCEPTS:
        return 1.0
    Ls = [R['L'] for R in ROT_ALL]
    return rot_memo('index', Ls, lambda: math.log2(max(2, int(torch.stack(
        [(torch.zeros(L.shape[:2], device=dev).scatter_(1, L.argmax(-1), 1.0) > 0).sum() for L in Ls]).sum()))))

def rot_index_bits_t():
    """rot_index_bits as a device scalar, with no host wait (0 under CONCEPTS: the gate's concept is rot_tau_bits')."""
    if CONCEPTS:
        return torch.zeros((), device=dev)
    Ls = [R['L'] for R in ROT_ALL]
    return rot_memo('index_t', Ls, lambda: torch.stack(
        [(torch.zeros(L.shape[:2], device=dev).scatter_(1, L.argmax(-1), 1.0) > 0).sum() for L in Ls]).sum().clamp_min(2).float().log2())

# torch.compile (speed): each map's per-token rotation and gate work runs as one function, compiled in the training
# pass on CUDA (inductor fuses the element-wise chains of the forward and of the backward); M's own products, the
# bases Q, the noise draws and the pass's records stay outside it. A core runs every product in full float32 (its
# matrix products read the TF32 switch when they run); calibration, evaluation and MPS run the same cores eagerly.
# On an A40 the element-wise work was about three quarters of a whole-model step's device time.
COMPILE = dev == 'cuda'
if COMPILE:
    import torch._dynamo, torch._inductor.config
    torch._dynamo.config.cache_size_limit = 64
    torch._inductor.config.fallback_random = True
_compiled = {}

def rot_run(core, *a):
    """core(*a) (its last argument the pass's mode) with every product in full float32, compiled in the training
    pass."""
    f = core
    if COMPILE and a[-1] == 'soft' and torch.is_grad_enabled() and state.get('calib') is None:
        f = _compiled.get(core) or _compiled.setdefault(core, torch.compile(core, dynamic=False))
    with full_float32():
        return f(*a)

def rot_core_args():
    """The pass's sequences forced on (a device index, or None): a core's input, so a compiled core never sees a
    changing Python number."""
    return (torch.tensor(state['force_on'], device=dev) if state['force_on'] else None,)

def rot_record(recs, Rbs, keys, Rs=None):
    """Appends a core's per-token records (bits on, blocks on, training-gate bits) and, in calibration, its blocks'
    reads to the pass's state; with the pin, each block set's reads and bits (Rs: the block sets)."""
    for i_, ((hb, n_on, sb, Lj, Hs), Rb, key) in enumerate(zip(recs, Rbs, keys)):
        if Hs is not None:
            state['gate_H'].append(Hs)
        if state['mode'] == 'hard':
            # A hard pass's third record is its blocks' gates [B, T, ng, g], kept for the probes.
            if state.get('probe') is not None:
                state['probe'][key] = (sb.detach(), Rs[i_] if Rs is not None else None)
            sb = None
        if state.get('calib') is not None:
            state['calib'].setdefault(key, []).append(Rb.detach().reshape(-1))
        if state.get('pin') is not None and Rs is not None:
            # The clean sequences' every eighth token.
            state['pin'].append((Rs[i_], Rb.detach().index_select(0, state['pin_rows'])[:, ::8], Lj.detach()))
        state['hard'].append(hb); state['rot_on'].append(n_on)
        if sb is not None:
            state['soft'].append(sb)

# DESCENT_LOGGATE=1: a block's guard reads its own write on a log scale, z = (ln R - tau) / s with tau a log threshold
# and s one e-fold (toys, cd27062d48): a block writing nothing is off exactly and the training noise is relative;
# with the linear read z = (R - tau) / s, tokens whose write is zero keep Phi(-tau / s) of every block in the
# expected bits.
LOGGATE = os.environ.get('DESCENT_LOGGATE') == '1'

def rot_gate_core(R, r, bits_i, idx, on, mode, extra=None):
    """The gates of R's slices [B, T, ng, g] from their reads r and their bits bits_i [ng, g] (extra: the gate
    network's term in z); with the per-token records (bits on, blocks on, training-gate bits or None) and the
    blocks' reads Rb."""
    g = R['L'].shape[-1]
    hot = (R['L'].argmax(-1, keepdim=True) == torch.arange(g, device=r.device)).float(); Lsm = torch.softmax(R['L'], -1)
    # Every pass runs the one-hot assignment (the delivered program's), with the softmax's gradient in training: the
    # expected assignment mixed blocks' gates per slice, so lev-st's training pass (hard gates) scored 3.3 bits per
    # token against its delivered program's 6.2. The guard's value is the one-hot read's and its gradient the
    # expected read's: at the one-hot read an empty block's read is 1e-10, whose gradient overflowed to NaN.
    guard = lambda R_: ((R_.log() if LOGGATE else R_) - R['tau']) / R['s']
    Rb = (qeinsum('...ni,nij->...nj', r.pow(2), hot) + 1e-20).sqrt()
    if mode == 'hard':
        M_ = hot
        z = guard(Rb)
    else:
        M_ = hot + Lsm - Lsm.detach()
        zs = guard((qeinsum('...ni,nij->...nj', r.pow(2), Lsm) + 1e-20).sqrt())
        z = guard(Rb.detach()).detach() + zs - zs.detach()
    if extra is not None:
        z = z + extra
    hard, phi = (z > 0).float() * R['keep'], 0.5 * (1 + torch.erf(z / SQ2)) * R['keep']
    if on is not None:
        hard, phi = hard.index_fill(0, on, 1.0), phi.index_fill(0, on, 1.0)
    Lj = qeinsum('ni,nij->nj', bits_i, M_) + rot_tau_bits(R) + idx
    hb, n_on = (hard * Lj).sum((-1, -2)).reshape(-1), hard.sum((-1, -2)).reshape(-1)
    if mode == 'hard':
        return qeinsum('...nj,nij->...ni', hard, hot), (hb, n_on, hard, Lj, None), Rb
    gb = rot_train_gate(hard, phi)
    Hs = None
    if GATE_H:
        # Each gate's binary entropy in bits, summed per token.
        ph = phi.clamp(1e-7, 1 - 1e-7)
        Hs = -(ph * ph.log2() + (1 - ph) * (1 - ph).log2()).sum((-1, -2)).reshape(-1)
    return qeinsum('...nj,nij->...ni', gb, M_), (hb, n_on, (gb * Lj).sum((-1, -2)).reshape(-1), Lj, Hs), Rb

def rot_fc_core(R, p, xe, Q, ex, idx, on, sc, mode):
    """A c_fc map after M's product p = x W^T: its slices' coefficients (with the read noise xe), its blocks' gates
    (ex: the gate network's term) and its output; returns (output, gates, records, block reads)."""
    sh = p.shape[:-1]
    mean_fc, mean_dn = R.get('mean_fc'), R.get('mean_dn')
    c = qeinsum('...nk,nki->...ni', permuted(p if mean_fc is None else p - mean_fc, R['perm'], R['inv']).view(*sh, R['ng'], ROTG), Q)
    if xe is not None:
        c = c + xe.view(*sh, R['ng'], ROTG) * R['ls_fc'].exp()
    recs, Rbs = [], []
    if mode == 'all':
        gam = torch.ones_like(c)
    else:
        bits_i, wn = rot_slice_bits(R, Q)
        pa = p if sc is None else p * sc
        abar = pa if READSIDE else vpd_model.gelu_tanh(pa)
        if mean_dn is not None and not READSIDE:
            abar = abar - mean_dn                                                   # the block writes act - E[act]
        abar = permuted(abar, R['perm'], R['inv']).view(*sh, R['ng'], ROTG)
        gam, rec, Rb = rot_gate_core(R, qeinsum('...nk,nki->...ni', abar, Q).abs() * wn, bits_i, idx, on, mode, ex)
        recs.append(rec); Rbs.append(Rb)
    out = permuted(qeinsum('...ni,nki->...nk', c * gam, Q).reshape(*sh, -1), R['inv'], R['perm'])
    return (out if mean_fc is None else out + mean_fc), gam, recs, Rbs

def make_rot_fc(n, l):
    R = ROT[l]; W = T.site(n).W
    def fwd(x):
        if state.get('dense') is not None:
            state['dense'][(l, 1)] = x
        p = x @ W.T
        if state['mode'] == 'M':
            return p
        # c_fc rows G scaled by 1 + a (an entry-wise neuron edit), on the shared compiler's side: the represented
        # output rows after the block projection (h = D Pi p), and in the gate's all-on read GELU(D p).
        sc = None
        for b, kind, G, a in state['entry'].get(n, ()):
            if kind == 'out':
                sc = torch.ones(p.shape[0], 1, p.shape[-1], device=p.device) if sc is None else sc
                sc[b, 0, G] = 1 + a
        Q = rot_Q(R)
        xe = x @ torch.randn(R['ng'] * ROTG, x.shape[-1], device=x.device).T if state.get('noise') else None
        ex = gate_net(l, 'mlp', x).view(*p.shape[:-1], R['ng'], R['g']) if GN and state['mode'] != 'all' else None
        mode = 'all' if state.get('allon_family') == 'mlp' else state['mode']
        out, gam, recs, Rbs = rot_run(rot_fc_core, R, p, xe, Q, ex, rot_index_bits_t(), *rot_core_args(), sc, mode)
        if sc is not None:
            out = out * sc
        rot_record(recs, Rbs, [n], [R])
        state['rot'][l] = (gam, Q)
        if state.get('alone') is not None and mode == 'hard':
            state['alone'][l] = (p.detach(), gam.detach(), Q.detach())
        return out
    return fwd

def acts_alone(l, p, gam, Q, stride=8, n_tok=256):
    """Layer l's MLP blocks (rot) acting alone: for each block on at a token, its effect with only it on (its slices'
    write W_dn,G Q_j Q_j^T GELU(Q_j Q_j^T p_G)) against its effect in context (the group's write with the token's
    blocks on less that without it), as 1 - sum ||alone - context||^2 / sum ||context||^2 in the output space over the
    (token, block on) pairs of every stride-th token (up to n_tok). The context is the block's group: GELU acts per
    neuron and the groups partition the neurons, so blocks of different groups never interact. Returns (the score,
    the pairs, the two sums)."""
    R = ROT[l]; g = R['g']
    grp_ = lambda t_: permuted(t_, R['perm'], R['inv']).view(-1, R['ng'], g)
    mf = grp_(R['mean_fc'][None])[:, :, None, :] if 'mean_fc' in R else 0.0                 # the mean parts (MEANPARTS)
    md = grp_(R['mean_dn'][None])[:, :, None, :] if 'mean_dn' in R else 0.0
    P_ = grp_(p.reshape(-1, p.shape[-1])[::stride][:n_tok] - (R['mean_fc'] if 'mean_fc' in R else 0.0))
    c = qeinsum('tnk,nki->tni', P_, Q)                                                       # [t, ng, i] slice coordinates
    gs = gam.reshape(-1, R['ng'], g)[::stride][:n_tok]                                       # the slices' (hard) gates
    memT = (R['L'].argmax(-1) [:, None, :] == torch.arange(g, device=dev)[None, :, None]).float()[None]   # [1, ng, j, i]
    Mo = qeinsum('nki,nkm,nmj->nij', Q, R['Gdn'], Q)                                         # the output metric
    def f(m):
        # The writes (slice coordinates) of the slices in masks m [t, ng, j, i], beside the mean parts'.
        return m * qeinsum('tnjk,nki->tnji', vpd_model.gelu_tanh(mf + qeinsum('tnji,nki->tnjk', m * c[:, :, None, :], Q)) - md, Q)
    ctx = f(gs[:, :, None, :]) - f(gs[:, :, None, :] * (1 - memT))
    alone = f(memT.expand(gs.shape[0], -1, -1, -1))
    on = (gs[:, :, None, :] * memT).amax(-1)                                                 # [t, ng, j]: on and non-empty
    nrm = lambda v: qeinsum('tnji,nik,tnjk->tnj', v, Mo, v)
    num, den = (nrm(alone - ctx) * on).sum(), (nrm(ctx) * on).sum()
    return 1 - (num / den.clamp_min(1e-30)).item(), int(on.sum().item()), num.item(), den.item()

def rot_dn_core(R, a, gam, Q, mode):
    """A down_proj map's slices on its input a, gated by its c_fc blocks' gates: (z before M's product, coefficients)."""
    sh = a.shape[:-1]
    mean_dn = R.get('mean_dn')
    c = qeinsum('...nk,nki->...ni', permuted(a if mean_dn is None else a - mean_dn, R['perm'], R['inv']).view(*sh, R['ng'], ROTG), Q) * gam
    z = permuted(qeinsum('...ni,nki->...nk', c, Q).reshape(*sh, -1), R['inv'], R['perm'])
    return (z if mean_dn is None else z + mean_dn), c

def make_rot_dn(n, l):
    R = ROT[l]; W = T.site(n).W
    def fwd(a):
        if state['mode'] == 'M':
            return a @ W.T
        sh = a.shape[:-1]
        gam, Q = state['rot'][l]
        for b, kind, G, a_ in state['entry'].get(n, ()):
            if kind == 'in':
                # down_proj columns G scaled by 1 + a (an entry-wise neuron edit), on the shared compiler's side: the
                # represented input columns before the block projection (y = W_dn Pi D a).
                sc = torch.ones(a.shape[-1], device=a.device); sc[G] = 1 + a_
                a = a.index_copy(0, torch.tensor([b], device=a.device), (a[b] * sc)[None])
        z, c = rot_run(rot_dn_core, R, a, gam, Q, state['mode'])
        y = z @ W.T
        if state.get('noise'):
            eps = torch.randn(R['ng'] * ROTG, W.shape[0], device=a.device)
            y = y + (c * R['ls_dn'].exp()).reshape(*sh, -1) @ eps
        return y
    return fwd

if ARM == 'rot':
    for l in range(T.n_layer):
        T.site(f'h.{l}.mlp.c_fc')._forward = make_rot_fc(f'h.{l}.mlp.c_fc', l)
        T.site(f'h.{l}.mlp.down_proj')._forward = make_rot_dn(f'h.{l}.mlp.down_proj', l)

# The rot arm's attention (DESCENT_SITES=all), exact by construction, from M's maps. q, k and the heads' value
# coordinates each fall into groups of DESCENT_ROT coordinates of one head, each group with a learned basis
# Q = exp(S) as in the MLP (pre-RoPE, so any basis sums to M's q and k), and a group's slices into blocks under a
# learned assignment, one gate per block:
# - q blocks, gated at the query t, switch their part of q(t); a block reads its effect on the logits there: the
#   root of its slices' summed variances of their logit terms c_i(t) (R_t Q_i) . k(s) / sqrt(HD) over the keys
#   under the all-on pattern (each slice's variance from the keys' per-coordinate variances, RoPE undone at t;
#   covariances between coordinates left out);
# - k blocks, gated at the key s, switch their part of k(s), reading their own write |c_i(s)|;
# - OV blocks: slice i reads W_v,G^T Q_i (mixed by the pattern) and writes W_o,G Q_i; a block's gate at the query t
#   switches its subspace of the head's output there (z_G = Q diag(g) Q^T a_G), reading its own write.
# With every block on, P's attention is M's. A q or k slice's description is its dense read (d numbers), an OV
# slice's its read and write, each at its own width under zero-mean priors per layer and map, plus its share of
# the rotation's angles.
ROTA = {}
if ARM == 'rot' and attn:
    NPL = HD // 2
    with torch.no_grad():
        for l in range(T.n_layer):
            Wq, Wk, Wv, Wo = (T.site(f'h.{l}.attn.{k}').W for k in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            GA = ROTGA or HD; ng = NH * HD // GA
            sub = lambda: {'ng': ng, 'g': GA, 'A': torch.zeros(ng, GA, GA, device=dev, requires_grad=True),
                           'L': (ASSIGN0(GA) * torch.eye(GA, device=dev)).repeat(ng, 1, 1).requires_grad_(),
                           'tau': torch.zeros(ng, GA, device=dev, requires_grad=True), 's': torch.ones(ng, GA, device=dev)}
            ROTA[l] = {}
            for name, W in (('q', Wq), ('k', Wk)):
                G_ = W.view(ng, GA, -1)
                ROTA[l][name] = {**sub(), 'G': G_ @ G_.transpose(1, 2), 'di': W.shape[1],
                                 'ls': torch.full((ng, GA), math.log(0.01 * W.pow(2).mean().sqrt().item()), device=dev, requires_grad=True)}
            Gv = Wv.view(ng, GA, -1); Go = Wo.T.reshape(ng, GA, -1)
            Q0s = {}
            if ATTN_START == 'vpd' and ng == NH:
                # Per-head groups: each head's basis starts at VPD's subcomponents' writes restricted to the head's
                # coordinates, largest ||v_i|| ||u_i,head|| first, orthonormalized in that order.
                for name, n_ in (('q', 'q_proj'), ('k', 'k_proj'), ('ov', 'v_proj')):
                    Vv_, Uv_ = load(f'h.{l}.attn.{n_}.V'), load(f'h.{l}.attn.{n_}.U')
                    Qh = []
                    for hh in range(NH):
                        Uh = Uv_[:, hh * HD:(hh + 1) * HD]
                        order = torch.argsort(-(Vv_.norm(dim=0) * Uh.norm(dim=1))).cpu().numpy()
                        D_ = (Uh / Uh.norm(dim=1, keepdim=True).clamp_min(1e-12)).T.cpu().double().numpy()[:, order]
                        Qh.append(torch.tensor(sl.qr(D_, mode='full')[0][:, :GA], dtype=torch.float32, device=dev))
                    Q0s[name] = torch.stack(Qh)
            if ATTN_START == 'vpd' and ng == 1:
                # VPD's subcomponents' directions as the start's basis: q's, k's and v's writes (in the map's output
                # space), largest ||v_i|| ||u_i|| first, orthonormalized in that order (completed by the orthogonal
                # complement), so the first slices start as VPD's largest subcomponents' directions; exact for any Q0.
                for name, n_ in (('q', 'q_proj'), ('k', 'k_proj'), ('ov', 'v_proj')):
                    Vv_, Uv_ = load(f'h.{l}.attn.{n_}.V'), load(f'h.{l}.attn.{n_}.U')          # [d_in, C], [C, d_out]
                    order = torch.argsort(-(Vv_.norm(dim=0) * Uv_.norm(dim=1))).cpu().numpy()
                    D_ = (Uv_ / Uv_.norm(dim=1, keepdim=True).clamp_min(1e-12)).T.cpu().double().numpy()[:, order]
                    Q0s[name] = torch.tensor(sl.qr(D_, mode='full')[0][:, :GA], dtype=torch.float32, device=dev)[None]
            ROTA[l]['ov'] = {**sub(), 'Gfc': Gv @ Gv.transpose(1, 2), 'Gdn': Go @ Go.transpose(1, 2), 'di': Wv.shape[1], 'do': Wo.shape[0],
                             'ls_fc': torch.full((ng, GA), math.log(0.01 * Wv.pow(2).mean().sqrt().item()), device=dev, requires_grad=True),
                             'ls_dn': torch.full((ng, GA), math.log(0.01 * Wo.pow(2).mean().sqrt().item()), device=dev, requires_grad=True)}
            for name, Q0 in Q0s.items():
                ROTA[l][name]['Q0'] = Q0
ROT_ALL = list(ROT.values()) + [R[x] for R in ROTA.values() for x in ('q', 'k', 'ov')]
for R in ROT_ALL:
    R['keep'] = torch.ones_like(R['s'])                                             # 0: pruned (never on)

def nonempty(R, soft=False):
    """Per block [ng, g]: 1 where a slice is assigned to it (soft: 1 - prod_i (1 - A_ij), A the assignment softmax)."""
    if soft:
        return 1 - torch.log1p(-torch.softmax(R['L'], -1).clamp_max(1 - 1e-6)).sum(1).exp()
    return torch.zeros(R['L'].shape[:2], device=dev).scatter_(1, R['L'].argmax(-1), 1.0)

def slices_live():
    """The slice arms' slices not pruned (threshold below 1e29), every map's."""
    return sum((P[n]['tau'] < 1e29).sum() for n in mlp) + sum((A[n]['tau'] < 1e29).sum() for n in sliced)

def gates_evaluated(soft):
    """CONCEPTS: the gates a reader evaluates on every token: each non-empty unpruned block's (soft: expected under the
    assignment softmax), the head scores the score gates read, and the gate networks' parameters."""
    extra = sum(t_.numel() for P_ in GN.values() for t_ in P_.values()) + sum(t_.numel() for t_ in TRUNK.values())
    extra += T.n_layer * NH if SCOREGATE and ROTA else 0
    extra += 2 * T.n_layer * NH if MEANQK and ROTA else 0
    extra += 2 * T.n_layer + (T.n_layer * NH if ROTA else 0) if MEANPARTS else 0
    if ARM != 'rot':
        # A slice arm's mean parts: one per MLP map and per head of every attention map that has one.
        extra = sum(1 for n in mlp if slice_mean(n)) + NH * sum(1 for n in sliced if slice_mean(n))
        return slices_live().float() + extra
    return sum((nonempty(R, soft) * R['keep']).sum() for R in ROT_ALL) + extra

def concepts_total():
    """CONCEPTS: the whole decomposition's concepts: each non-empty unpruned block's 1 + 4 per MLP or OV slice (3 per q or
    k slice), plus the gates' extra concepts (head scores, gate network parameters)."""
    if ARM != 'rot':
        return float(gates_evaluated(False) + (CPS - 1) * slices_live())
    tot = gates_evaluated(False) - sum((nonempty(R) * R['keep']).sum() for R in ROT_ALL)
    for R in ROT_ALL:
        ranks = torch.zeros(R['L'].shape[:2], device=dev).scatter_add_(1, R['L'].argmax(-1), torch.ones(R['L'].shape[:2], device=dev))
        tot = tot + ((ranks > 0).float() * R['keep'] * (1 + (3 if 'G' in R else 4) * ranks)).sum()
    return float(tot)

# DESCENT_GATENET=H: each layer's gates also read a small causal network of the layer's own input at the token
# (the normed residual stream entering the attention, or the MLP): one hidden layer of H units (GELU), its output
# added to every block's z there. Linear own reads cannot decode a superposed code; the network can. Its
# weights are part of the switching function, described once per run under F (zero-mean priors), not counted per
# token; the output layer starts at zero, so the start is the own-read gates'.
GATENET = int(os.environ.get('DESCENT_GATENET', '0'))
# DESCENT_GATENET_DENSE=1: the gate networks read, at each token, the streams of every layer from a dense pass run
# first (every part on: P = M up to the parts' sum), VPD's own_causal_1 scheme (its causal CI network reads every
# site's input from such a pass); without it each reads only its own layer's stream from the gated pass.
GN_DENSE = os.environ.get('DESCENT_GATENET_DENSE') == '1'
# DESCENT_GATECTX=D (with DESCENT_GATENET_DENSE): the dense streams first pass through a shared trunk, a projection to
# D dimensions and one causal attention layer over the tokens (4 heads), so every gate also sees the context before
# the token, as VPD's causal CI network does; the gate networks read the trunk's output.
GATECTX = int(os.environ.get('DESCENT_GATECTX', '0'))
TRUNK = {}

def gate_trunk(f):
    """The shared trunk on the dense streams f [B, T, d_f]: h = f W_in, then h + causal self-attention of h."""
    h = f @ TRUNK['Win']
    B_, T_, D_ = h.shape; nh = 4
    sp = lambda t_: t_.view(B_, T_, nh, D_ // nh).transpose(1, 2)
    q_, k_, v_ = sp(h @ TRUNK['Wq']), sp(h @ TRUNK['Wk']), sp(h @ TRUNK['Wv'])
    o = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True).transpose(1, 2).reshape(B_, T_, D_)
    return h + o @ TRUNK['Wo']
GN = {}
if ARM == 'rot' and GATENET:
    for l in range(T.n_layer):
        outs = {'mlp': ROT[l]['ng'] * ROT[l]['g']}
        if ROTA:
            outs['attn'] = sum(ROTA[l][x]['ng'] * ROTA[l][x]['g'] for x in ('q', 'k', 'ov'))
        for part, n_out in outs.items():
            d_ = GATECTX or T.wte.shape[1] * ((2 if ROTA else 1) * T.n_layer if GN_DENSE else 1)
            GN[(l, part)] = {'W1': (torch.randn(d_, GATENET, device=dev) * math.sqrt(2 / d_)).requires_grad_(),
                             'b1': torch.zeros(GATENET, device=dev, requires_grad=True),
                             'W2': torch.zeros(GATENET, n_out, device=dev, requires_grad=True)}

if ARM != 'rot' and GATENET:
    # Slice arms: one network per map, reading its layer's stream (the MLP's or the attention's normed input), one
    # output per slice.
    for n in mlp + sliced:
        n_out = P[n]['V'].shape[1] if n in P else NH * A[n]['V'].shape[-1]
        d_ = GATECTX or T.wte.shape[1] * ((2 if sliced else 1) * T.n_layer if GN_DENSE else 1)
        GN[n] = {'W1': (torch.randn(d_, GATENET, device=dev) * math.sqrt(2 / d_)).requires_grad_(),
                 'b1': torch.zeros(GATENET, device=dev, requires_grad=True),
                 'W2': torch.zeros(GATENET, n_out, device=dev, requires_grad=True)}

if GATECTX and GN_DENSE and GN:
    d_f = T.wte.shape[1] * (2 if (ROTA or sliced) else 1) * T.n_layer
    TRUNK.update({'Win': (torch.randn(d_f, GATECTX, device=dev) / math.sqrt(d_f)).requires_grad_(),
                  **{k: (torch.randn(GATECTX, GATECTX, device=dev) / math.sqrt(GATECTX)).requires_grad_() for k in ('Wq', 'Wk', 'Wv', 'Wo')}})

def gate_net_s(n, x):
    """Map n's gate network's output for each of its slices at each token of x [B, T, d] (with GN_DENSE, of the
    dense pass's streams at every layer instead)."""
    P_ = GN[n]
    if GN_DENSE:
        x = state['gn_feats']
    return F.gelu(x @ P_['W1'] + P_['b1']) @ P_['W2']

def gate_net(l, part, x):
    """The layer's gate network's output for every block of `part` at each token of x [B, T, d] (zero without one)."""
    if (l, part) not in GN:
        return 0.0
    P_ = GN[(l, part)]
    if GN_DENSE:
        x = state['gn_feats']
    return F.gelu(x @ P_['W1'] + P_['b1']) @ P_['W2']

def rot_read_bits(R, Q):
    """Per q or k slice [ng, g]: its dense read's description and its angles' share, in bits."""
    if CONCEPTS:
        return torch.full_like(R['s'], 3.0)
    s2 = (2 * R['ls']).exp(); d_ = R['di']
    nr = qeinsum('nki,nkm,nmi->ni', Q, R['G'], Q)
    v = (nr.sum() + d_ * s2.sum()) / (d_ * nr.numel())
    return (0.5 * (d_ * torch.log(v / s2) + (nr + d_ * s2) / v - d_)) / math.log(2) + rot_angle_bits(R)

def rot_attention_core(Rq, Rk, Ro, q, k, v, causal, Qq, Qk, Q, noise, ex, idx, on, mode):
    """Layer's attention after M's q, k, v products: q blocks gated at the query, k blocks at the key, OV blocks on
    the heads' output at the query (ex: the gate network's terms for q, k and OV, or None); returns (the input of M's o_proj, the OV coefficients, records, block reads)."""
    B_, T_ = q.shape[0], q.shape[1]
    grp = lambda t_: t_.view(B_, T_, -1, Rq['g'])
    mean_q, mean_k = Rq.get('mean'), Rk.get('mean')
    if mean_q is not None:
        q, k = q - mean_q, k - mean_k
    cq = qeinsum('btnk,nki->btni', grp(q), Qq); ck = qeinsum('btnk,nki->btni', grp(k), Qk)
    if noise is not None:
        cq = cq + grp(noise[0]) * Rq['ls'].exp()
        ck = ck + grp(noise[1]) * Rk['ls'].exp()
    rope = lambda t_: T._rope(t_.view(B_, T_, NH, HD).transpose(1, 2), T_)
    back = lambda c_, Q_: qeinsum('btni,nki->btnk', c_, Q_).reshape(B_, T_, -1)
    bq = lambda c_: back(c_, Qq) if mean_q is None else back(c_, Qq) + mean_q             # with the positional parts
    bk = lambda c_: back(c_, Qk) if mean_k is None else back(c_, Qk) + mean_k
    recs, Rbs = [], []
    if mode != 'all':
        # The keys' per-coordinate variance over the attended keys, RoPE undone at the query: for plane c,
        # cov_rel = R_t^T cov_c(t) R_t; coordinate c (first of its plane) takes cov_rel[0, 0], c + HD/2 cov_rel[1, 1].
        with torch.no_grad():
            qh, kh = rope(bq(cq)), rope(bk(ck))
            full = ((qh @ kh.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(~causal, float('-inf')).softmax(-1)
        k2 = torch.stack((kh[..., :NPL], kh[..., NPL:]), -1)                                    # [B, H, T, NPL, 2]
        mk = (full @ k2.flatten(-2)).view(B_, NH, T_, NPL, 2)
        kk = (full @ (k2[..., :, None] * k2[..., None, :]).flatten(-3)).view(B_, NH, T_, NPL, 2, 2)
        cv = kk - mk[..., :, None] * mk[..., None, :]
        co, si = T.cos[:T_, :NPL], T.sin[:T_, :NPL]
        d0 = co ** 2 * cv[..., 0, 0] + 2 * co * si * cv[..., 0, 1] + si ** 2 * cv[..., 1, 1]
        d1 = si ** 2 * cv[..., 0, 0] - 2 * co * si * cv[..., 0, 1] + co ** 2 * cv[..., 1, 1]
        D = torch.cat((d0, d1), -1).clamp_min(0).transpose(1, 2).reshape(B_, T_, -1, Rq['g'])  # [B, T, ng, g] by coordinate
        rq = (cq.pow(2) * qeinsum('btnk,nki->btni', D, Qq.pow(2)) / HD + 1e-20).sqrt()
        gq, rec, Rb = rot_gate_core(Rq, rq, rot_read_bits(Rq, Qq), idx, on, mode, ex and ex[0]); recs.append(rec); Rbs.append(Rb)
        gk, rec, Rb = rot_gate_core(Rk, ck.abs(), rot_read_bits(Rk, Qk), idx, on, mode, ex and ex[1]); recs.append(rec); Rbs.append(Rb)
        cq, ck = cq * gq, ck * gk
    qh, kh = rope(bq(cq)), rope(bk(ck))
    pattern = ((qh @ kh.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(~causal, float('-inf')).softmax(-1)
    a = (pattern @ v.view(B_, T_, NH, HD).transpose(1, 2)).transpose(1, 2).reshape(B_, T_, -1)
    mean_v = Ro.get('mean')
    a = grp(a if mean_v is None else a - mean_v)
    c = qeinsum('btnk,nki->btni', a, Q)
    if noise is not None:
        # Each OV slice's read noise at each key, mixed by the pattern like the values.
        vn = noise[2].view(B_, T_, NH, HD).transpose(1, 2)
        c = c + grp((pattern @ vn).transpose(1, 2).reshape(B_, T_, -1)) * Ro['ls_fc'].exp()
    if mode != 'all':
        bits_i, wn = rot_slice_bits(Ro, Q)
        go, rec, Rb = rot_gate_core(Ro, c.abs() * wn, bits_i, idx, on, mode, ex and ex[2]); recs.append(rec); Rbs.append(Rb)
        c = c * go
    return (back(c, Q) if mean_v is None else back(c, Q) + mean_v), c, recs, Rbs

def rot_attention(i, h, causal):
    """Layer i's attention output under the rot arm: M's q, k, v and o products around rot_attention_core."""
    Rq, Rk, Ro = ROTA[i]['q'], ROTA[i]['k'], ROTA[i]['ov']; B_, T_ = h.shape[0], h.shape[1]
    if state.get('dense') is not None:
        state['dense'][(i, 0)] = h
    site = lambda k: T.site(f'h.{i}.attn.{k}')
    q, k, v = site('q_proj')(h), site('k_proj')(h), site('v_proj')(h)
    Qq, Qk, Q = rot_Q(Rq), rot_Q(Rk), rot_Q(Ro)
    noise = None
    if state.get('noise'):
        noise = (h @ torch.randn(q.shape[-1], h.shape[-1], device=h.device).T, h @ torch.randn(k.shape[-1], h.shape[-1], device=h.device).T,
                 h @ torch.randn(NH * HD, h.shape[-1], device=h.device).T)
    ex = None
    if GN and state['mode'] != 'all':
        # The gate network's output for every q, k and OV block of the layer, split by map.
        gn, ex, o_ = gate_net(i, 'attn', h).view(B_, T_, -1), [], 0
        for R_ in (Rq, Rk, Ro):
            m_ = R_['ng'] * R_['g']; ex.append(gn[..., o_:o_ + m_].view(B_, T_, R_['ng'], R_['g'])); o_ += m_
        ex = tuple(ex)
    if SCOREGATE and state['mode'] != 'all':
        # Each head's largest attention logit at the token (M's q and k on P's stream), standardized per head on
        # the first pass, read by the q, k and OV blocks' gates (z += beta s).
        with torch.no_grad():
            sm = ((T._rope(q.view(B_, T_, NH, HD).transpose(1, 2), T_) @ T._rope(k.view(B_, T_, NH, HD).transpose(1, 2), T_).transpose(-1, -2))
                  / math.sqrt(HD)).masked_fill(~causal, float('-inf')).amax(-1)                  # [B, H, T]
            if i not in SCORE_NORM:
                SCORE_NORM[i] = (sm.mean((0, 2))[None, :, None], sm.std((0, 2)).clamp_min(1e-6)[None, :, None])
            sc = ((sm - SCORE_NORM[i][0]) / SCORE_NORM[i][1]).transpose(1, 2)                   # [B, T, H]
        if 'beta' in Rq:                                                            # (the weights come after calibration)
            terms = [sc[..., None] * R_['beta'] if R_['ng'] == NH else qeinsum('bth,ngh->btng', sc, R_['beta']) for R_ in (Rq, Rk, Ro)]
            ex = tuple(terms) if ex is None else tuple(e_ + t_ for e_, t_ in zip(ex, terms))
    z, c, recs, Rbs = rot_run(rot_attention_core, Rq, Rk, Ro, q, k, v, causal, Qq, Qk, Q, noise, ex, rot_index_bits_t(),
                              *rot_core_args(), 'all' if state.get('allon_family') == 'attn' else state['mode'])
    rot_record(recs, Rbs, [f'h.{i}.attn.q_proj', f'h.{i}.attn.k_proj', f'h.{i}.attn.o_proj'], [Rq, Rk, Ro])
    y = site('o_proj')(z)
    if state.get('noise'):
        y = y + (c * Ro['ls_dn'].exp()).reshape(B_, T_, -1) @ torch.randn(NH * HD, y.shape[-1], device=h.device)
    return y

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
    l = int(g.integers(T.n_layer)); W = lambda k: T.site(site_of(l, k)).W
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
    """edits: per sequence of the batch, None, 'allon' (every part's gate on, against M), or
    {map: (A, B), '_entry': {map: (kind, G, a)}}: a weight edit, additive Delta W = A B^T on M, and on P
    entry-wise where '_entry' gives it (kind 'in': input coordinates G of the map scaled by 1 + a, in
    every part's read; 'out': output coordinates G, in every part's write), additive otherwise."""
    state['wedits_M'], state['wedits_P'], state['entry'] = {}, {}, {}
    state['force_on'] = []
    for b, e in enumerate(edits):
        if e == 'allon':
            state['force_on'].append(b); continue
        entry = (e or {}).get('_entry', {})
        for n, v in (e or {}).items():
            if n == '_entry':
                continue
            state['wedits_M'].setdefault(n, []).append((b, *v))
            if n in entry and (n in P or n in A):
                state['entry'].setdefault(n, []).append((b, *entry[n]))
            else:
                state['wedits_P'].setdefault(n, []).append((b, *v))

# DESCENT_WEDITS edited sequences per 8 of the batch (the same fraction at every batch size).
N_EDITS = int(os.environ.get('DESCENT_WEDITS', '0')) * batch // 8
# DESCENT_ALLON=1: one sequence per 8 of the batch runs with every part's gate on, against M (ties the parts'
# sum to M).
ALLON = int(os.environ.get('DESCENT_ALLON', '0') == '1') * max(1, batch // 8)
# DESCENT_REPEATS=n: n clean sequences of every 8 are a chunk of text followed by the same chunk again, so the training
# text holds repeats, where the model copies by induction (Pile text rarely does: the rot explanation scored 4.67
# bits per token on a repeat against 2.45 on the first pass at 10M tokens, VPD's switching function 1.32).
REPEATS = int(os.environ.get('DESCENT_REPEATS', '0')) * batch // 8
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
    mg = cmv = None
    if state['mode'] != 'all' and slice_mean(n):
        # The mean part: the pattern's rows sum to 1, so the mixed coefficients less the mean coefficients are the
        # mixed deviations.
        cmv = head_coefficients(XBAR[n][None], p['V'], False)[None]               # [1, H, 1, C]
        m = m - cmv
    if state['mode'] == 'all':
        g = 1.0; mg = m * g
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
            ex = gate_net_s(n, h).view(B_, T_, NH, -1).permute(0, 2, 1, 3) if n in GN else None
            if 'beta' in p:
                sg = state['score'][l][..., None] * p['beta'][None, :, None, :]
                ex = sg if ex is None else ex + sg
            args = (r, m, p['U'].norm(dim=-1)[None, :, None, :], p['tau'][None, :, None, :], p['s'][None, :, None, :],
                    p['taun'][None, :, None, :] if 'taun' in p else None, ex)
            if not state['force_on']:
                mg, hb, sb, hd = slice_apply_run(*args, (1, 3), state['mode'])
                if hd is not None:
                    state['on'][n] = hd
            else:
                hard, soft = slice_gates_run(*args)
        if mg is not None:
            state['hard'].append(hb.reshape(-1))
            if sb is not None:
                state['soft'].append(sb.reshape(-1))
        else:
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
            mg = m * g
    y = (mg if cmv is None else mg + cmv) @ p['U'][None]                       # [B, H, T, HD]
    for b, Ae, Be in state['wedits_P'].get(n, ()):
        # A weight edit of v_proj adds Delta W x to the values, which the pattern mixes like any value.
        dv = ((h[b] @ Be) @ Ae.T).view(T_, NH, HD).transpose(0, 1)               # [H, T, HD]
        y = y.index_add(0, torch.tensor([b], device=y.device), (pattern[b] @ dv)[None])
    return y.transpose(1, 2).reshape(B_, T_, -1)

def rope_rot(w):
    """rotate_half of the model's RoPE: rope_u(w) = w cos_u + rope_rot(w) sin_u."""
    n_ = HD // 2
    return torch.cat([-w[..., n_:], w[..., :n_]], -1)

def dest_k_scores(l, h, q, causal):
    """DESTKV: layer l's attention scores [B, H, T, T] with each k slice gated at the query: s_tu = sum over slices c
    on at t of c(u) (q_t . rope_u(u_c)) / sqrt(hd), plus the mean key part's term (MEANQK), over each query's
    candidates (every slice on; in training also those with z > -2, for their gradient)."""
    n = f'h.{l}.attn.k_proj'; p = A[n]
    B_, T_ = h.shape[0], h.shape[1]
    c = head_coefficients(h.reshape(-1, h.shape[-1]), p['V'], False).view(NH, B_, T_, -1).permute(1, 0, 2, 3)   # [B, H, T, C]
    U_ = p['U']                                                                  # [H, C, HD]
    C_ = U_.shape[1]
    kbar = None
    if slice_mean(n):
        cm = head_coefficients(XBAR[n][None], p['V'], False)                    # [H, 1, C]
        c = c - cm[None]
        kbar = cm @ U_                                                           # [H, 1, HD], the mean key part
    Qr = q / math.sqrt(HD)
    cos, sin = T.cos[:T_].T.contiguous(), T.sin[:T_].T.contiguous()             # [HD, T]
    with torch.no_grad():
        # Each slice's signal at each query: max over u <= t of |c(u) (Qr_t . rope_u(u_c))|, TF32 (gate input only).
        sig = torch.empty(B_, NH, T_, C_, device=h.device)
        cc = max(1, int(2e8 // (B_ * NH * T_ * T_)))
        Kr = lambda U_c: (U_c[:, :, None, :] * cos.T[None, None] + rope_rot(U_c)[:, :, None, :] * sin.T[None, None])   # [H, cc, T, HD]
        for c0 in range(0, C_, cc):
            Kc = Kr(U_[:, c0:c0 + cc])
            sc_ = torch.einsum('bhtd,hcud->bhctu', Qr.detach(), Kc) * c[:, :, :, c0:c0 + cc].detach().permute(0, 1, 3, 2)[:, :, :, None, :]
            sig[..., c0:c0 + cc] = sc_.abs_().masked_fill_(~causal, 0).amax(-1).permute(0, 1, 3, 2)
    if state.get('calib') is not None:
        state['calib'].setdefault(n, []).append(sig)
    z = (sig - p['tau'][None, :, None, :]) / p['s'][None, :, None, :]
    hard, phi = (z > 0).float(), 0.5 * (1 + torch.erf(z / SQ2))
    if state['force_on']:
        on = torch.tensor(state['force_on'], device=hard.device)
        hard, phi = hard.index_fill(0, on, 1.0), phi.index_fill(0, on, 1.0)
        z = z.index_fill(0, on, 1e9)
    state['hard'].append(hard.sum((1, 3)).reshape(-1))
    if state['mode'] == 'hard':
        g = hard
        state['on'][n] = hard
    else:
        state['soft'].append(phi.sum((1, 3)).reshape(-1))
        g = phi if gate == 'mf' else hard + phi - phi.detach()
    n_sel = int((z > (0.0 if state['mode'] == 'hard' else -2.0)).sum(-1).max())
    K = min(C_, 1 << max(3, math.ceil(math.log2(max(n_sel, 1)))))
    cT = c.transpose(2, 3)                                                       # [B, H, C, T]
    ck = lambda f, *a: torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False) if torch.is_grad_enabled() else f(*a)
    parts = []
    if K * 8 >= C_:
        # Many on: every slice, by RoPE's form, s_tu = sum_d cos_ud sum_c g_c(t) Qr_td u_cd c(u) + the sin term.
        Ut, Rt = U_.transpose(1, 2)[None, :, None], rope_rot(U_).transpose(1, 2)[None, :, None]   # [1, H, 1, HD, C]
        def dense(Qb, gb):
            X = Qb[..., None] * gb[..., None, :]                                 # [B, H, tb, HD, C]
            return (((X * Ut) @ cT[:, :, None]) * cos).sum(3) + (((X * Rt) @ cT[:, :, None]) * sin).sum(3)
        tb = max(1, int(1e8 // (B_ * NH * HD * max(C_, T_))))
        for t0 in range(0, T_, tb):
            parts.append(ck(dense, Qr[:, :, t0:t0 + tb], g[:, :, t0:t0 + tb]))
    else:
        # Few on: each query's K candidates (every slice on; in training also those with z > -2, for their gradient).
        idx = z.detach().topk(K, -1).indices                                     # [B, H, T, K]
        gs = g.gather(-1, idx)
        bi = torch.arange(B_, device=h.device)[:, None, None, None]; hi = torch.arange(NH, device=h.device)[None, :, None, None]
        def sparse(Qb, gb, ib):
            W = U_[hi, ib]                                                       # [B, H, tb, K, HD]
            Qt = Qb[:, :, :, None, :]
            beta = (Qt * W) @ cos + (Qt * rope_rot(W)) @ sin                     # [B, H, tb, K, T]
            return (beta * cT[bi, hi, ib] * gb[..., None]).sum(3)                # [B, H, tb, T]
        tb = max(1, int(1e8 // (B_ * NH * K * T_)))
        for t0 in range(0, T_, tb):
            parts.append(ck(sparse, Qr[:, :, t0:t0 + tb], gs[:, :, t0:t0 + tb], idx[:, :, t0:t0 + tb]))
    s_ = torch.cat(parts, 2)
    if kbar is not None:
        s_ = s_ + Qr @ T._rope(kbar[None].expand(B_, -1, T_, -1), T_).transpose(-1, -2)
    return s_

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
        if ROTA:
            x = x + rot_attention(i, h, causal)
        else:
            if SCOREGATE and state['mode'] != 'all':
                with torch.no_grad():
                    qf = T._rope((h @ site('q_proj').W.T).view(B_, T_, NH, HD).transpose(1, 2), T_)
                    kf = T._rope((h @ site('k_proj').W.T).view(B_, T_, NH, HD).transpose(1, 2), T_)
                    sm = ((qf @ kf.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(~causal, float('-inf')).amax(-1)   # [B, H, T]
                    if i not in SCORE_NORM:
                        SCORE_NORM[i] = (sm.mean((0, 2))[None, :, None], sm.std((0, 2)).clamp_min(1e-6)[None, :, None])
                    state['score'][i] = (sm - SCORE_NORM[i][0]) / SCORE_NORM[i][1]
            q = T._rope(site('q_proj')(h).view(B_, T_, NH, HD).transpose(1, 2), T_)
            if DESTKV and state['mode'] != 'all':
                pattern = dest_k_scores(i, h, q, causal).masked_fill(~causal, float('-inf')).softmax(-1)
            else:
                k = T._rope(site('k_proj')(h).view(B_, T_, NH, HD).transpose(1, 2), T_)
                pattern = ((q @ k.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(~causal, float('-inf')).softmax(-1)
            x = x + site('o_proj')(attn_v(i, h, pattern))
        h = vpd_model.rms(x, T.norms[2 * i + 1], T.eps)
        x = x + site('down_proj')(vpd_model.gelu_tanh(site('c_fc')(h)))
    return vpd_model.rms(x, T.ln_f, T.eps)
if attn:
    T.hidden = hidden
if sliced:
    # Calibration on P's own run with every slice on (= M): each v slice's noise scale and the v map's
    # threshold from the post-attention reads (the quantile matching VPD's mean count at the map).
    with torch.no_grad():
        state['capture'] = {}
        for i in range(0, 4, 2):
            state['mode'], state['soft'], state['hard'] = 'all', [], []
            T(torch.tensor(tok[i:i + 2, :512].astype(np.int64), device=dev))
        cap = {n: torch.cat(v, 1) for n, v in state['capture'].items()}
        state['capture'] = None
        for l in range(T.n_layer):
            n = f'h.{l}.attn.v_proj'; r = cap[n]
            A[n]['s'] = 0.1 * r.pow(2).mean(1).sqrt().clamp_min(1e-12)
            q = start_q(1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / r.shape[0] / r.shape[2])
            flat = r.reshape(-1); idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            A[n]['tau'] = torch.full_like(A[n]['s'], torch.quantile(flat[idx], q).item()).requires_grad_()
        if ARM == 'share':
            for l in range(T.n_layer):
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
    return KLBits.apply(lm, lp)

def kl_rows(m, p):
    """KL(softmax(m) || softmax(p)) in bits per row of logits [n, V]."""
    pm = F.log_softmax(m.float(), -1); pp = F.log_softmax(p.float(), -1)
    return (pm.exp() * (pm - pp)).sum(-1) / math.log(2)

def kl_rows_grad(m, p, g):
    """kl_rows' gradient in p, times g [n]: (softmax(p) - softmax(m)) g / ln 2."""
    return (F.softmax(p.float(), -1) - F.softmax(m.float(), -1)) * (g[:, None] / math.log(2))

class KLBits(torch.autograd.Function):
    """KL(M || P) in bits per token from M's logits lm and P's lp [..., V], over chunks of tokens, its gradient in lp
    taken directly, (softmax(lp) - softmax(lm)) / ln 2: only the two logit tensors are kept for the backward
    (autograd through log_softmax kept five vocabulary-wide tensors; the whole model ran out of an A40's memory at
    32 x 512 tokens). On CUDA the row functions are compiled: a few passes over the logits in place of eleven."""
    CHUNK = 2048
    rows, rows_grad = (torch.compile(kl_rows, dynamic=False), torch.compile(kl_rows_grad, dynamic=False)) if COMPILE else (kl_rows, kl_rows_grad)
    @staticmethod
    def forward(ctx, lm, lp):
        ctx.save_for_backward(lm, lp)
        V = lm.shape[-1]; fm, fp = lm.reshape(-1, V), lp.reshape(-1, V)
        out = torch.empty(fm.shape[0], device=lm.device)
        for i in range(0, fm.shape[0], KLBits.CHUNK):
            out[i:i + KLBits.CHUNK] = KLBits.rows(fm[i:i + KLBits.CHUNK], fp[i:i + KLBits.CHUNK])
        return out.view(lm.shape[:-1])
    @staticmethod
    def backward(ctx, g):
        lm, lp = ctx.saved_tensors
        V = lm.shape[-1]; fm, fp, fg = lm.reshape(-1, V), lp.reshape(-1, V), g.reshape(-1)
        grad = torch.empty_like(fp)
        for i in range(0, fm.shape[0], KLBits.CHUNK):
            j = slice(i, i + KLBits.CHUNK)
            grad[j] = KLBits.rows_grad(fm[j], fp[j], fg[j])
        return None, grad.view(lp.shape)

def run(ids, mode):
    if GN_DENSE and mode in ('soft', 'hard'):
        # The two-pass switching function: a dense pass first (every part on), whose streams at every layer
        # (the attention's and the MLP's normed inputs) the gate networks read at each token.
        with torch.no_grad():
            state['mode'], state['soft'], state['hard'], state['edges_soft'], state['edges_hard'] = 'all', [], [], [], []
            state['rot'], state['rot_on'], state['dense'] = {}, [], {}
            T(ids)
            state['gn_feats'] = torch.cat([state['dense'][k] for k in sorted(state['dense'])], -1)
            state['dense'] = None
        if TRUNK:
            state['gn_feats'] = gate_trunk(state['gn_feats'])
    state['mode'], state['soft'], state['hard'], state['edges_soft'], state['edges_hard'] = mode, [], [], [], []
    state['rot'], state['rot_on'], state['gate_H'] = {}, [], []
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
def evaluate(final=False, start_eval=False):
    if EXACT:
        for n in mlp:
            if not (NEURON_DOWN and n.endswith('down_proj')):
                P[n]['V'], P[n]['U'] = frame(P[n]['F'], T.site(n).W)
    for n in sliced if not ATTN_FREE else ():
        A[n]['V'], A[n]['U'] = head_frame(A[n]['F'], T.site(n).W, A[n]['o'])
    state['collect'] = {} if SHARE else None
    install([None])
    r = {'kl': [], 'kl_soft': [], 'kl_all_on': [], 'active': [], 'active_soft': [], 'per_map': [], 'edges_on': [], 'edges_on_soft': []}
    graph = None
    if CONCEPTS:
        G_ev, seen, seen_s = float(gates_evaluated(False)), {}, {}
        state['on'] = {}
    for i in range(0, ev.shape[0], 4):
        ids = ev[i:i + 4]
        lm = run(ids, 'M')
        state['probe'] = {} if CONCEPTS else None
        state['alone'] = {} if ROT and i == 0 else None
        lp = run(ids, 'hard'); r['kl'].append(kl_bits(lm, lp).mean().item())
        if state['alone'] is not None:
            # The MLP blocks acting alone (on the first held-out sequences).
            alone_rec = {l: acts_alone(l, *v) for l, v in state['alone'].items()}
            state['alone'] = None
        r['active'].append(torch.stack(state['hard']).sum(0).mean().item())
        if CONCEPTS and ARM != 'rot':
            # Each slice's on-anywhere (for the pruning): MLP [B, T, C], q/k/o [H, tokens, C], v [B, H, T, C].
            for n_, hd_ in state['on'].items():
                a_ = hd_.amax((0, 1)) if n_ in P else hd_.amax(1) if hd_.dim() == 3 else hd_.amax((0, 2))
                seen_s[n_] = torch.maximum(seen_s[n_], a_) if n_ in seen_s else a_
        if CONCEPTS and ARM == 'rot':
            # Each block's on-anywhere (for the pruning).
            for g_, R_ in state['probe'].values():
                seen[id(R_)] = (R_, torch.maximum(seen[id(R_)][1], g_.amax((0, 1))) if id(R_) in seen else g_.amax((0, 1)))
            state['probe'] = None
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
        if ROTA:
            # The hard error split: the attention blocks gated with every MLP block on, and the reverse.
            for fam, key in (('mlp', 'kl_attn_gated'), ('attn', 'kl_mlp_gated')):
                state['allon_family'] = fam
                lp = run(ids, 'hard'); r.setdefault(key, []).append(kl_bits(lm, lp).mean().item())
            state['allon_family'] = None
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
    # Induction: 16 random 100-token sequences, each followed by itself again. KL(M || P) on the first and on the second
    # copy, M's and P's loss in bits on the second copy (each of its tokens can be copied from the first), and (rot
    # with attention) the blocks on at the second copy's tokens against the first's, with each block's map and rank.
    g_ = torch.Generator().manual_seed(5)
    ind = torch.randint(0, T.wte.shape[0], (16, 100), generator=g_).to(dev).repeat(1, 2)
    install([None] * 16)
    lm = run(ind, 'M')
    state['probe'] = {} if (ROT and ROTA) or CONCEPTS else None
    lp = run(ind, 'hard')
    kl_t = kl_bits(lm, lp)
    loss = lambda lg: (-F.log_softmax(lg[:, 100:-1].float(), -1).gather(-1, ind[:, 101:, None]).mean() / math.log(2)).item()
    rows = []
    for key, (gates, R) in (state['probe'] or {}).items():
        if CONCEPTS:
            seen[id(R)] = (R, torch.maximum(seen[id(R)][1], gates.amax((0, 1))) if id(R) in seen else gates.amax((0, 1)))
        g1, g2 = gates[:, 1:100].mean((0, 1)), gates[:, 100:].mean((0, 1))                 # [ng, g] on-rates
        members = torch.zeros_like(g1).scatter_add_(1, R['L'].argmax(-1), torch.ones_like(g1)) if R is not None else g1 * 0
        for n_, j_ in zip(*torch.nonzero((g2 - g1) > 0.5, as_tuple=True)):
            rows.append({'map': key, 'group': int(n_), 'block': int(j_), 'rank': int(members[n_, j_]),
                         'on_repeat': round(g2[n_, j_].item(), 3), 'on_first': round(g1[n_, j_].item(), 3)})
    state['probe'] = None
    out['induction'] = {'kl_first': round(kl_t[:, 1:100].mean().item(), 3), 'kl_repeat': round(kl_t[:, 100:].mean().item(), 3),
                        'loss_repeat_M': round(loss(lm), 3), 'loss_repeat_P': round(loss(lp), 3),
                        'blocks_on_repeat_not_first': sorted(rows, key=lambda r: -(r['on_repeat'] - r['on_first']))[:40]}
    if ROT:
        out['mlp_acts_alone'] = {l: round(v[0], 4) for l, v in alone_rec.items()}
        out['mlp_acts_alone_all'] = round(1 - sum(v[2] for v in alone_rec.values()) / max(1e-30, sum(v[3] for v in alone_rec.values())), 4)
    if CONCEPTS:
        # The delivered program's concepts per token (on, plus the gates evaluated); then the non-empty blocks never on
        # in these hard passes (held-out text, the induction check) are pruned.
        out['gates_evaluated'] = G_ev
        out['concepts_on'] = CPS * out['active']
        out['concepts_per_token'] = CPS * out['active'] + G_ev
        cut_n = 0
        for R_, anyon in (seen.values() if not start_eval else ()):
            # (none at the start's evaluation: the start's thresholds are a calibration, not a trained choice)
            cut = (anyon == 0) & (nonempty(R_) > 0) & (R_['keep'] > 0)
            R_['keep'][cut] = 0.0; cut_n += int(cut.sum())
        with torch.no_grad():
            for n_, anyon in (seen_s.items() if not start_eval else ()):
                # A slice arm's slice never on is pruned: its threshold out of reach (it stays in every-part-on).
                cont = P[n_] if n_ in P else A[n_]
                cut = (anyon == 0) & (cont['tau'] < 1e29)
                cont['tau'][cut] = 1e30; cut_n += int(cut.sum())
        out['pruned_now'] = cut_n
        out['pruned_total'] = (int(sum(((R_['keep'] == 0) & (nonempty(R_) > 0)).sum() for R_ in ROT_ALL)) if ARM == 'rot'
                               else int(sum((P[n]['tau'] >= 1e29).sum() for n in mlp) + sum((A[n]['tau'] >= 1e29).sum() for n in sliced)))
        out['concepts_total'] = concepts_total()
    if ROT:
        run(ev[0:4], 'hard')
        ranks = torch.cat([torch.bincount(R['L'].argmax(-1).reshape(-1) + ROTG * torch.arange(R['ng'], device=dev).repeat_interleave(ROTG),
                                          minlength=R['ng'] * ROTG) for R in ROT.values()])
        ranks = ranks[ranks > 0].float()
        rk = lambda L: torch.cat([torch.bincount((L.argmax(-1) + L.shape[-1] * torch.arange(L.shape[0], device=dev)[:, None]).reshape(-1),
                                                 minlength=L.shape[0] * L.shape[-1])]).float()
        if ROTA:
            for name in ('q', 'k', 'ov'):
                r_ = torch.cat([rk(R[name]['L']) for R in ROTA.values()]); r_ = r_[r_ > 0]
                out.setdefault('rot_attn', {})[name] = {'blocks': int(r_.numel()), 'rank_hist': {str(k): int((r_ == k).sum()) for k in range(1, 65) if (r_ == k).any()}}
        out['rot'] = {'blocks_on': round(torch.stack(state['rot_on']).sum(0).mean().item(), 2), 'blocks': int(ranks.numel()),
                      'rank_hist': {str(k): int((ranks == k).sum()) for k in range(1, ROTG + 1) if (ranks == k).any()},
                      'mean_slice_bits': round(float(np.mean([rot_slice_bits(R, rot_Q(R))[0].mean().item() for R in ROT.values()])), 1),
                      'rotation_angle_rms': round(float(np.mean([(R['A'] * MASK).pow(2).sum().div(MASK.sum() * R['ng']).sqrt().item() for R in ROT.values()])), 4),
                      'width_rel_fc': round(float(np.mean([R['ls_fc'].exp().mean().item() / T.site(f'h.{l}.mlp.c_fc').W.pow(2).mean().sqrt().item() for l, R in ROT.items()])), 4)}
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

# The VPD or neuron start's thresholds set in the gated run: each map's threshold is the quantile of its reads that
# matches its target count (VPD's per-map count scaled to the budget) with every upstream map gated at its
# own threshold, one pass per map in forward order (after pass j the first j maps are at their fixed
# point). Thresholds set on M's inputs left maps dead whose reads shrink under upstream gating (the
# whole model at K = 128: layer 0's v and o and layer 1's o at 0.0-0.1 on against 1.2-3.2 targeted, and a
# gate far below its threshold gets no gradient back).
# DESCENT_RESUME=SAVE: the parts and gates start from a DESCENT_SAVE file (the posterior means; the optimizer,
# posterior widths and multiplier start afresh), its thresholds kept. Under F the weight priors stay centred
# on the start's tensors (M's own slices for the neuron and head starts), recorded before the load.
RESUME = os.environ.get('DESCENT_RESUME')
START_VAL = {}
if RESUME:
    S_ = torch.load(RESUME, map_location=dev, weights_only=False)
    with torch.no_grad():
        for cont, src in ([(P[n], S_['maps'][n]) for n in mlp] + [(A[n], S_['attn'][n]) for n in sliced]
                          ):
            for k in cont:
                if k in src and torch.is_tensor(cont[k]) and k != 'F':
                    START_VAL[(id(cont), k)] = cont[k].detach().clone()
                    cont[k] = src[k].to(dev).clone().requires_grad_(k != 's')
    print('resumed from', RESUME, 'step', S_['step'], flush=True)
if start in ('vpd', 'neuron') and not (SHARE or SHARE_A or EXACT or ARM in ('dir', 'rot') or RESUME):
    # A neuron part counts its two slices: its layer's neurons on match the mean of VPD's two counts.
    vc = lambda n: (VPD_COUNTS[n] if start == 'vpd' else
                    (VPD_COUNTS[n.rsplit('.', 1)[0] + '.c_fc'] + VPD_COUNTS[n.rsplit('.', 1)[0] + '.down_proj']) / 2)
    target = {n: 1 - START_SCALE * vc(n) / P[n]['V'].shape[1] for n in mlp}
    target.update({n: 1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / (NH * A[n]['V'].shape[-1]) for n in sliced})
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

if DESTKV and sliced:
    # The k slices' thresholds on their destination signal: the start's quantile of it (as the own reads' were set),
    # the noise scale a tenth of its root mean square per slice. And a check: with every k slice on, the scores are
    # M's.
    with torch.no_grad():
        ids_k = torch.tensor(tok[0:2, :512].astype(np.int64), device=dev)
        install([None]); state['calib'] = {}
        run(ids_k, 'soft')
        for l in range(T.n_layer):
            n = f'h.{l}.attn.k_proj'
            sg = torch.stack(state['calib'][n]).view(-1, NH, 512, A[n]['tau'].shape[-1])
            A[n]['s'] = 0.1 * sg.pow(2).mean((0, 2)).sqrt().clamp_min(1e-12)
            flat = sg.reshape(-1); idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            A[n]['tau'].fill_(torch.quantile(flat[idx], start_q(1 - START_SCALE * VPD_ATTN_COUNTS.get(n, 1.0) / (NH * sg.shape[-1]))).item())
        state['calib'] = None
        install(['allon', 'allon']); state['mode'] = 'soft'
        h0 = vpd_model.rms(T.wte[ids_k], T.norms[0], T.eps)
        q0 = T._rope((h0 @ T.site('h.0.attn.q_proj').W.T).view(2, 512, NH, HD).transpose(1, 2), 512)
        k0 = T._rope((h0 @ T.site('h.0.attn.k_proj').W.T).view(2, 512, NH, HD).transpose(1, 2), 512)
        causal0 = torch.ones(512, 512, dtype=torch.bool, device=dev).tril()
        s_d, s_m = dest_k_scores(0, h0, q0, causal0), (q0 @ k0.transpose(-1, -2)) / math.sqrt(HD)
        print('destkv check: every k slice on, scores against M\'s, max |diff|', float((s_d - s_m).masked_fill(~causal0, 0).abs().max()),
              'of', float(s_m.masked_fill(~causal0, 0).abs().max()), flush=True)
        install([None])

if ARM == 'rot':
    # B: the bits of K of VPD's MLP subcomponents, each at a width of 1% of its tensor's root mean square under the
    # zero-mean prior per tensor, plus its index among VPD's MLP subcomponents.
    with torch.no_grad():
        vb, total = {}, 0
        for n in mlp + attn:
            per = 0.0
            for w, ax in (('V', 0), ('U', 1)):
                t_ = load(f'{n}.{w}'); d_ = t_.shape[ax]
                s2 = (0.01 * t_.pow(2).mean().sqrt()) ** 2; v_ = t_.pow(2).mean() + s2
                per = per + 0.5 * (d_ * torch.log(v_ / s2) + (t_.pow(2).sum(ax) + d_ * s2) / v_ - d_) / math.log(2)
            vb[n] = per.mean().item(); total += per.numel()
        counts = {**VPD_COUNTS, **(VPD_ATTN_COUNTS if attn else {})}
        K = K / sum(counts.values()) * sum(counts[n] * (vb[n] + math.log2(total)) for n in counts)
        if CONCEPTS:
            K = KAPPA
    print('rot budget B', round(K), 'bits per token; VPD subcomponent bits', {n: round(v) for n, v in vb.items()}, flush=True)
    # Each block's noise scale (a tenth of the root mean square of its read with all on), then the thresholds set
    # in the gated run, layer by layer, so the start's expected bits per token are B (the layers' shares as VPD's
    # MLP counts there).
    ids_c = torch.tensor(tok[0:4, :512].astype(np.int64), device=dev)
    # Components in forward order: (blocks' container, threshold, noise scale, read key, start bits per block, share).
    comps = []
    for l in range(T.n_layer):
        if ROTA:
            for name, key, bits0, share in (
                    ('q', 'q_proj', lambda R: rot_read_bits(R, rot_Q(R)), VPD_ATTN_COUNTS[f'h.{l}.attn.q_proj']),
                    ('k', 'k_proj', lambda R: rot_read_bits(R, rot_Q(R)), VPD_ATTN_COUNTS[f'h.{l}.attn.k_proj']),
                    ('ov', 'o_proj', lambda R: rot_slice_bits(R, rot_Q(R))[0],
                     VPD_ATTN_COUNTS[f'h.{l}.attn.v_proj'] + VPD_ATTN_COUNTS[f'h.{l}.attn.o_proj'])):
                R = ROTA[l][name]
                comps.append((R, 'tau', 's', f'h.{l}.attn.{key}', lambda R=R, f=bits0: f(R).mean().item(), share))
        R = ROT[l]
        comps.append((R, 'tau', 's', f'h.{l}.mlp.c_fc', lambda R=R: rot_slice_bits(R, rot_Q(R))[0].mean().item(),
                      VPD_COUNTS[f'h.{l}.mlp.c_fc'] + VPD_COUNTS[f'h.{l}.mlp.down_proj']))
    if MEANQK or MEANPARTS:
        # The mean parts on the calibration text, from the attention's and the MLP's inputs with all on (M's): each
        # head's mean q and k (pre-RoPE), mean value, and each MLP's mean pre-activation and activation.
        with torch.no_grad():
            install([None]); state['dense'] = {}
            run(ids_c, 'all')
            for l in range(T.n_layer):
                if ROTA:
                    h_ = state['dense'][(l, 0)].reshape(-1, T.wte.shape[1])
                    for name, key, on_ in (('q', 'q_proj', MEANQK), ('k', 'k_proj', MEANQK), ('ov', 'v_proj', MEANPARTS)):
                        if on_:
                            ROTA[l][name]['mean'] = (h_ @ T.site(f'h.{l}.attn.{key}').W.T).mean(0)
                if MEANPARTS:
                    p_ = state['dense'][(l, 1)].reshape(-1, T.wte.shape[1]) @ T.site(f'h.{l}.mlp.c_fc').W.T
                    ROT[l]['mean_fc'], ROT[l]['mean_dn'] = p_.mean(0), vpd_model.gelu_tanh(p_).mean(0)
            state['dense'] = None
    with torch.no_grad():
        install([None])
        state['calib'] = {}
        run(ids_c, 'soft')
        for R, tk, sk, key, _, _ in comps:
            rb = torch.stack(state['calib'][key]).view(-1, *R[sk].shape)
            R[sk] = torch.ones_like(R[sk]) if LOGGATE else 0.1 * rb.pow(2).mean(0).sqrt().clamp_min(1e-12)
        ib = rot_index_bits(); share_all = sum(c[5] for c in comps)
        for R, tk, sk, key, bits0, share in comps:
            state['calib'] = {}
            run(ids_c, 'hard')
            flat = torch.cat(state['calib'][key])
            on = K * share / share_all / (bits0() + ib)
            idx = torch.randperm(flat.numel(), device=dev)[:2_000_000]
            fl = flat[idx].float()
            R[tk].fill_(torch.quantile(fl.log() if LOGGATE else fl, 0.01 if CONCEPTS else max(0.0, 1 - on / R[sk].numel())).item())
        state['calib'] = None
        run(ids_c, 'hard')
        print('rot start: bits per token', round(torch.stack(state['hard']).sum(0).mean().item()), 'blocks on per token',
              round(torch.stack(state['rot_on']).sum(0).mean().item(), 1), flush=True)

# Every trained tensor as (container, key, scale): Adam steps of 0.3% of the scale per step, the
# scale a weight tensor's root mean square and a threshold's its map's noise scale x 33. DESCENT_LR multiplies every step size (default 1), and every step
# size scales with the square root of the tokens per step against 8 x 256 (the batch's gradient noise).
LR = float(os.environ.get('DESCENT_LR', '1')) * math.sqrt(batch * seq / (8 * 256))
rms = lambda q: q.detach().pow(2).mean().sqrt().item()
# DESCENT_GATES_ONLY=1: the slices stay as they start (VPD's subcomponents, for the vpd start) and only the gates
# train: whether the gap to VPD is the gates or the parts (VPD's own causal gates on these parts: 0.80 bits per
# token at 190 active, whole model).
GATES_ONLY = os.environ.get('DESCENT_GATES_ONLY') == '1'
if GATES_ONLY:
    # The fixed parts take no gradient: no backward products for their reads and writes, no optimizer state.
    for cont in [P[n] for n in mlp] + [A[n] for n in sliced]:
        for w in ('V', 'U', 'F', 'G'):
            if torch.is_tensor(cont.get(w)) and cont[w].is_leaf:
                cont[w].requires_grad_(False)
# DESCENT_SIGNED=1 (slice arms): a slice's guard reads its signed own write, on when c ||u|| > tau or -c ||u|| > tau_neg,
# each sign its own threshold (both started at the calibrated |c| ||u|| threshold, so the start is unchanged):
# the read |c| ||u|| fires on a large negative pre-activation that GELU discards, and VPD's own gates on its parts
# are functions of the signed read.
if os.environ.get('DESCENT_SIGNED') == '1' and ARM != 'rot':
    for cont in [P[n] for n in mlp] + [A[n] for n in sliced]:
        cont['taun'] = cont['tau'].detach().clone().requires_grad_()
slots = [(P[n], w, rms(P[n][w])) for n in mlp for w in (('F',) if EXACT else ('V', 'U')) + (('G',) if ARM == 'dir' else ())
         if not (EXACT and NEURON_DOWN and n.endswith('down_proj') and w == 'F') and ARM != 'rot' and not GATES_ONLY]
# rot: rotation angles by 1e-3 per step, thresholds by a tenth of their noise scale, assignment logits by 0.02.
slots += [x for R in ROT_ALL for x in ((R, 'A', 1 / 3), (R, 'tau', 100 / 3 * R['s'].mean().item()), (R, 'L', 20 / 3))]
slots += [x for P_ in GN.values() for x in ((P_, 'W1', rms(P_['W1'])), (P_, 'b1', 0.1), (P_, 'W2', 1 / math.sqrt(GATENET)))]
if SCOREGATE:
    # beta by its threshold's step in z (a tenth of the noise scale per step, against a standardized score). A rot
    # attention block in a head's group reads that head's score; one in a group across heads reads every head's.
    for n in sliced:
        A[n]['beta'] = torch.zeros_like(A[n]['tau']).detach().requires_grad_()
    slots += [(A[n], 'beta', 100 / 3) for n in sliced]
    for R in [R_[x] for R_ in ROTA.values() for x in ('q', 'k', 'ov')]:
        R['beta'] = torch.zeros(R['ng'], R['g'], *(() if R['ng'] == NH else (NH,)), device=dev, requires_grad=True)
        slots.append((R, 'beta', 100 / 3))
slots += [(TRUNK, k, rms(TRUNK[k])) for k in TRUNK]
slots += [(A[n], w, rms(A[n][w])) for n in sliced for w in (('V', 'U') if ATTN_FREE else ('F',)) if not GATES_ONLY] + [(A[n], 'tau', 100 / 3 * A[n]['s'].mean().item()) for n in sliced]
for l, S in SHARE_A.items():
    slots += [(S, 't', 100 / 3 * S['s'].mean().item()), (S, 'L_v', 20 / 3), (S, 'L_o', 20 / 3)]
slots += [(P[n], 'tau', 100 / 3 * P[n]['s'].mean().item()) for n in mlp if ARM != 'rot']
slots += [(cont, 'taun', 100 / 3 * cont['s'].mean().item()) for cont in [P[n] for n in mlp] + [A[n] for n in sliced] if 'taun' in cont]
for l, S in SHARE.items():
    # part thresholds by 1% of their noise scale x 10, assignment logits by 0.02 per step
    slots += [(S, 't', 100 / 3 * S['s'].mean().item()), (S, 'L_fc', 20 / 3), (S, 'L_dn', 20 / 3)]
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
if FMODE:
    # rot: each slice's dense widths (log sigma by 1% per step), and the leaves its cost reads.
    groups += [{'params': [R[w] for w in ('ls_fc', 'ls_dn', 'ls') if w in R], 'lr': LR * 1e-2} for R in ROT_ALL]
    for cont, key, mu, ls in leaves:
        if any(cont is R for R in ROT_ALL) and key in ('A', 'tau'):
            cont[key + '_leaf'] = (mu, ls)
class OneAdam(torch.optim.Adam):
    """torch's multi-tensor Adam (its default on CUDA) in one pass over every group instead of one per group
    (each tensor here is its own group: the per-group passes held the host 24 ms of a whole-model step on an
    A40). The same element-wise operations, each tensor at its group's step size."""
    @torch.no_grad()
    def step(self):
        ps, gs, ms, vs, ss, lrs = [], [], [], [], [], []
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['step'] = torch.tensor(0.0)
                    st['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                ps.append(p); gs.append(p.grad); ms.append(st['exp_avg']); vs.append(st['exp_avg_sq']); ss.append(st['step']); lrs.append(g['lr'])
        if not ps:
            return
        (b1, b2), eps = self.defaults['betas'], self.defaults['eps']
        torch._foreach_add_(ss, torch.tensor(1.0), alpha=1.0)
        torch._foreach_lerp_(ms, gs, 1 - b1)
        torch._foreach_mul_(vs, b2)
        torch._foreach_addcmul_(vs, gs, gs, 1 - b2)
        bc1 = [1 - b1 ** s_.item() for s_ in ss]
        bc2 = [1 - b2 ** s_.item() for s_ in ss]
        den = torch._foreach_sqrt(vs)
        torch._foreach_div_(den, [bc ** 0.5 for bc in bc2])
        torch._foreach_add_(den, eps)
        torch._foreach_addcdiv_(ps, ms, den, [(lr_ / bc) * -1 for lr_, bc in zip(lrs, bc1)])
opt = OneAdam(groups)
trainable = [q for g in groups for q in g['params']]
# DESCENT_LR_DECAY=1: every step size falls linearly to a tenth of its start over the run's steps. At a constant
# step the whole model's held-out KL rose from 2.78 to 2.91 bits per token between 10M and 15M tokens (and
# descent-rotvpd's stalled at 3.0 from 10M), the gradient noise holding it off its minimum.
LR_DECAY = os.environ.get('DESCENT_LR_DECAY') == '1'
for g_ in groups:
    g_['lr0'] = g_['lr']

def draw(mean):
    """Install the posterior mean (mean) or one sample of every tensor (F only), then (exact) every
    map's reads and writes from its frame."""
    for cont, key, mu, ls in leaves:
        cont[key] = mu if mean else mu + ls.exp() * torch.randn_like(mu)
    state['noise'] = FMODE and not mean and ARM == 'rot'
    if EXACT:
        for n in mlp:
            if not (NEURON_DOWN and n.endswith('down_proj')):
                P[n]['V'], P[n]['U'] = frame(P[n]['F'], T.site(n).W)
    for n in sliced if not ATTN_FREE else ():
        A[n]['V'], A[n]['U'] = head_frame(A[n]['F'], T.site(n).W, A[n]['o'])


def description_bits():
    """KL(q || p) in bits (F only)."""
    total = 0.0
    for R in ROT_ALL:
        if 'ls' in R:
            # A q or k slice's dense read.
            s2 = (2 * R['ls']).exp(); d_ = R['di']
            Q = rot_Q({'A': R['A_leaf'][0], **({'Q0': R['Q0']} if 'Q0' in R else {})}) if 'A_leaf' in R else rot_Q(R)
            nr = qeinsum('nki,nkm,nmi->ni', Q, R['G'], Q)
            v = (nr.sum() + d_ * s2.sum()) / (d_ * nr.numel())
            total = total + 0.5 * (d_ * torch.log(v / s2) + (nr + d_ * s2) / v - d_).sum()
            continue
        # rot: every slice's dense read and write (the angles and thresholds are leaves below).
        sf, sd = (2 * R['ls_fc']).exp(), (2 * R['ls_dn']).exp()
        Q = rot_Q({'A': R['A_leaf'][0], **({'Q0': R['Q0']} if 'Q0' in R else {})}) if 'A_leaf' in R else rot_Q(R)
        nf = qeinsum('nki,nkm,nmi->ni', Q, R['Gfc'], Q); nd = qeinsum('nki,nkm,nmi->ni', Q, R['Gdn'], Q)
        for nn_, s2, d_ in ((nf, sf, R['di']), (nd, sd, R['do'])):
            v = (nn_.sum() + d_ * s2.sum()) / (d_ * nn_.numel())
            total = total + 0.5 * (d_ * torch.log(v / s2) + (nn_ + d_ * s2) / v - d_).sum()
    for cont, key, mu, ls in leaves:
        e = mu.pow(2) + (2 * ls).exp()
        if key == 'A' and any(cont is R for R in ROT_ALL):
            # Rotation angles: the strict upper triangle only.
            M_ = mask_of(mu.shape[-1])
            v = (e * M_).sum() / (M_.sum() * mu.shape[0])
            total = total + 0.5 * ((torch.log(v) - 2 * ls + e / v - 1) * M_).sum()
            continue
        if key in ('tau', 't', 'taun'):
            # Thresholds and widths are locations, not zero-centred: their prior N(m, v) has its mean m fitted
            # too (described in (1/2) ln n nats), v = mean((mu - m)^2 + sigma^2) at its optimum. Under
            # N(0, v) a tensor's thresholds, all near one value tau_0, had v ~ tau_0^2, which pushed every
            # sigma toward tau_0 (sigma / RMS(mu) 0.53 after 1,220 steps on the whole model) and fired
            # random gates at the sampled parameters.
            v = ((mu - mu.mean()).pow(2) + (2 * ls).exp()).mean()
            total = total + 0.5 * (mu.numel() * torch.log(v) - 2 * ls.sum()) + 0.5 * math.log(mu.numel())
            continue
        v = e.mean()
        total = total + 0.5 * (mu.numel() * torch.log(v) - 2 * ls.sum())
    return total / math.log(2)
# The budget: each step descends F + lam (E[k] - K); DESCENT_DUAL picks how lam (and the thresholds) follow it.
DUAL = os.environ.get('DESCENT_DUAL', 'logint')
if DUAL not in ('logint', 'pin', 'loghard'):
    raise SystemExit('DESCENT_DUAL: logint, pin or loghard')
# DESCENT_DUAL=loghard: logint whose lambda follows the delivered program's bits (a hard pass at the posterior mean
# on up to four of the step's clean sequences) instead of the training pass's: under logint the whole model's
# held-out hard program spent 2.96M of K = 4.22M bits per token at 10M tokens, the training count (sampled
# parameters and gates) sitting above it.
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
                    'attn': {n: {k: A[n][k].detach().float().cpu() for k in ('V', 'U', 'tau', 's', 'beta') if k in A[n]} for n in sliced},
                    'tied': {dn: (fc, own.cpu()) for dn, (fc, own) in GROUP.items()},
                    # rot: per layer the neuron order (groups of DESCENT_ROT consecutive), angles, assignments,
                    # thresholds, noise scales and the slices' log widths.
                    'rot': {l: {k: R[k].detach().float().cpu() if R[k].dtype.is_floating_point else R[k].cpu()
                                for k in ('perm', 'A', 'L', 'tau', 's', 'ls_fc', 'ls_dn', 'Q0', 'keep', 'mean_fc', 'mean_dn') if k in R} for l, R in ROT.items()},
                    # Each group's basis: Q = Q0 (I + S)^-1 (I - S) for groups larger than 64, Q0 exp(S) otherwise (Q0 the
                    # start's basis where saved, else I), S the skew part of A's strict upper triangle; the guard reads
                    # ln R against tau (log) or R (linear); the gate also adds beta . (the heads' standardized largest
                    # attention logits) where beta is saved, and is off where keep is 0 (pruned).
                    'rot_basis': {'cayley_above': 64, 'guard': 'log' if LOGGATE else 'linear'},
                    # The score gates' per-layer standardization of each head's largest attention logit (mean, std).
                    'score_norm': {l: (m_.flatten().cpu(), s_.flatten().cpu()) for l, (m_, s_) in SCORE_NORM.items()},
                    # rot attention: per layer the OV groups' (consecutive value coordinates of the heads) angles,
                    # assignments, thresholds, noise scales and widths, and the QK planes' assignments, thresholds,
                    # noise scales and widths.
                    'rota': {l: {x: {k: R[x][k].detach().float().cpu() for k in ('A', 'L', 'tau', 's', 'ls', 'ls_fc', 'ls_dn', 'Q0', 'keep', 'beta', 'mean') if k in R[x]}
                                 for x in ('q', 'k', 'ov')} for l, R in ROTA.items()},
                    # The gate networks (per layer and part for rot, per map otherwise; W1, b1, W2) and their input:
                    # 'dense' (every layer's streams from the dense pass) or the layer's own stream; the trunk if any.
                    'gate_nets': {str(k): {w: v.detach().float().cpu() for w, v in P_.items()} for k, P_ in GN.items()},
                    'gate_net_input': 'dense' if GN_DENSE else 'own', 'gate_trunk': {k: v.detach().float().cpu() for k, v in TRUNK.items()}},
                   os.environ['DESCENT_SAVE'])

draw(True)
e = evaluate(start_eval=True); e['weight_edits'] = evaluate_edits(); print('start', e, flush=True); log['trace'].append({'step': 0, **e}); save(0)
g_edits = np.random.default_rng(11)
step_seconds = []
t0 = time.time()
# DESCENT_DUAL=pin (toys, cd27062d48): lambda stays at the balance rate it starts at, and after every step every
# threshold moves by one common shift, the least for which the hard program's bits per clean token are at most K
# (bisection over the block reads of a hard pass at the posterior mean, the program delivered, on up to four of the
# step's clean sequences; no shift where the budget never binds). Pinning the expected (soft) count instead drove
# toys' hard KL to 158 bits per token, and the multiplier alone left the MLP 16% under budget; pinning the training
# pass's own count (sampled parameters and gates, upstream blocks drawn off) let the held-out hard program run
# 7.0M bits per token at 5M tokens against K = 4.2M.
def delivered_kl(ids, kinds, lm):
    """The delivered (hard) program's KL in bits per token on up to four of the step's clean natural-text sequences (not
    the repeated ones; lm: M's logits)."""
    rows = [b for b, k_ in enumerate(kinds) if k_ is None and b < len(kinds) - REPEATS][:4] or [0]
    install([None] * len(rows))
    with torch.no_grad():
        kl_ = kl_bits(lm[rows], run(ids[rows], 'hard')).mean().item()
    install(kinds)
    return kl_

def delivered_bits(ids, kinds):
    """The delivered (hard, posterior-mean) program's bits per token on up to four of the step's clean sequences."""
    rows = [b for b, k_ in enumerate(kinds) if k_ is None][:4] or [0]
    draw(True)
    install([None] * len(rows))
    with torch.no_grad():
        run(ids[rows], 'hard')
    return torch.stack(state['hard']).sum(0).mean().item()

def pin_shift(ids, kinds):
    rows = [b for b, k_ in enumerate(kinds) if k_ is None][:4]
    if not rows:
        return
    draw(True)
    install([None] * len(rows))
    state['pin'], state['pin_rows'] = [], torch.arange(len(rows), device=dev)
    with torch.no_grad():
        run(ids[rows], 'hard')
    pins, state['pin'] = state['pin'], None
    if not pins:
        return
    sets = [(R, (Rb.log() if LOGGATE else Rb).reshape(-1, *Lj.shape), Lj) for R, Rb, Lj in pins]
    def bits(d):
        tot = 0.0
        for R, rb, Lj in sets:
            tau = R['tau_leaf'][0] if 'tau_leaf' in R else R['tau']
            tot = tot + (((rb - tau.detach() - d) / R['s'] > 0).float() * Lj).sum((-1, -2)).mean()
        return float(tot)
    lo, hi = -20.0, 20.0
    if bits(lo) <= K:
        return
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if bits(mid) <= K else (mid, hi)
    with torch.no_grad():
        for R in {id(R): R for R, _, _ in sets}.values():
            (R['tau_leaf'][0] if 'tau_leaf' in R else R['tau']).add_(hi)

LOG_EVERY = max(1, 400_000 // (batch * seq))
with open(out.replace('.json', '.tsv'), 'w') as f_:
    f_.write('step\ttokens\ttrain_kl\ttrain_F\tbits_train_gates\tbits_hard\tlambda' + ('\tgates_evaluated\tkl_delivered' if CONCEPTS else '') + '\n')
# DESCENT_FRESH=1 (default under CONCEPTS): the training sequences are the token file's rows in a fixed random order,
# each used once (none repeated before the file's 134M tokens are used), the held-out rows 1024..1031 left out.
FRESH = os.environ.get('DESCENT_FRESH', '1' if CONCEPTS else '0') == '1'
if FRESH:
    order = np.random.default_rng(1).permutation(tok.shape[0] - 8); order = np.where(order >= 1024, order + 8, order)
for step in range(steps):
    rows = rng.integers(0, train_rows, batch); rows = np.where(rows >= 1024, rows + 8, rows); offs = rng.integers(0, 513 - seq, batch)
    if FRESH:
        rows = order[np.arange(step * batch, (step + 1) * batch) % len(order)]
    ids = torch.tensor(np.stack([tok[r, o:o + seq] for r, o in zip(rows, offs)]).astype(np.int64), device=dev)
    if REPEATS:
        rb = torch.arange(batch - REPEATS, batch, device=dev)                         # the batch's last (clean) sequences
        ids[rb, seq // 2:] = ids[rb, :seq - seq // 2]
    t_step = time.time()
    kinds = [draw_edit(g_edits, FAMILIES[(step * N_EDITS + b) % len(FAMILIES)]) if b < N_EDITS else None for b in range(batch)]
    for b in range(N_EDITS, N_EDITS + ALLON):
        kinds[b] = 'allon'
    install(kinds)
    with torch.no_grad():
        lm = run(ids, 'M')
    draw(False)
    state['pin'] = None
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
    ek = (torch.stack(state['soft']).sum(0) * counted).sum() / counted.sum()
    hk = ((torch.stack(state['hard']).sum(0) * counted).sum() / counted.sum()).item()
    train_hard = list(state['hard'])                    # the training pass's records (the pin's pass replaces state's)
    if EDGES:
        # The per-token budget counts the edges on beside the parts on.
        ek = ek + torch.stack(state['edges_soft']).sum(0).mean()
        hk += torch.stack(state['edges_hard']).sum(0).mean().item()
    if GATE_H:
        # (in bits, beside the KL)
        gate_H = torch.stack(state['gate_H']).sum(0).mean()
        objective = objective + gate_H
    if CONCEPTS:
        # The concepts per clean token (the training gates' blocks on, plus the gates evaluated), against mu times the
        # bits (KL and the gates' entropy). The delivered KL on the step's clean sequences drives mu: the training
        # pass's own under st (one-hot assignments, hard gates: the delivered program), else a hard pass.
        G_t = gates_evaluated(True)
        conc = CPS * ek + G_t
        # (natural text only: the repeated sequences train the copying, but the limit is on natural text, where
        # including them held the held-out KL at 0.63 against kappa 0.553)
        clean = [b for b, k_ in enumerate(kinds) if k_ is None and b < batch - REPEATS]
        kl_dl = kl_seq[clean].mean().item() if gate == 'st' else delivered_kl(ids, kinds, lm)
        if step == 0:
            # mu starts at the balance ||d concepts / d tau|| / ||d bits / d tau|| over all thresholds (a per-threshold
            # median overflowed: at the all-on start most thresholds move the KL by nearly nothing).
            taus = [cont[key] for cont, key, _ in slots if key == 'tau']
            gC = torch.autograd.grad(conc, taus, retain_graph=True, allow_unused=True)
            gB = torch.autograd.grad(objective, taus, retain_graph=True, allow_unused=True)
            nC = math.sqrt(sum(a.double().pow(2).sum().item() for a in gC if a is not None))
            nB = math.sqrt(sum(b.double().pow(2).sum().item() for b in gB if b is not None))
            lam = nC / nB if nB > 0 and math.isfinite(nC / nB) else 1.0
            print('mu start', lam, flush=True)
    if step == 0 and not CONCEPTS:
        # lambda starts at the median over thresholds of the balance |dF/dtau| / |dE[k]/dtau|.
        taus = [mu for _, key, mu, _ in leaves if key == 'tau'] if FMODE else [cont[key] for cont, key, _ in slots if key == 'tau']
        gF = torch.autograd.grad(objective, taus, retain_graph=True, allow_unused=True)
        gk = torch.autograd.grad(ek, taus, retain_graph=True, allow_unused=True)
        ratio = torch.cat([(a.abs() / b.abs()).reshape(-1)[b.abs().reshape(-1) > 0] for a, b in zip(gF, gk) if a is not None and b is not None])
        lam = max(ratio.median().item(), 1e-12) if ratio.numel() else 1e-3
    if LR_DECAY:
        for g_ in opt.param_groups:
            g_['lr'] = g_['lr0'] * (1 - 0.9 * step / max(1, steps - 1))
    opt.zero_grad(); (conc + lam * objective if CONCEPTS else objective + lam * (ek - K)).backward(); opt.step()
    if CONCEPTS:
        # mu: log-integral ascent on the delivered KL's relative excess over kappa (capped at +1, as below).
        lam = lam * math.exp(min((kl_dl - KAPPA) / KAPPA, 1.0) / B_H)
    elif DUAL == 'pin':
        pin_shift(ids, kinds)
    elif DUAL == 'loghard':
        lam = lam * math.exp(min((delivered_bits(ids, kinds) - K) / K, 1.0) / B_H)
    else:
        # The relative violation, capped at +1 as it is bounded by -1 below, so lambda rises no faster than
        # it can fall (an uncapped rise ran away at K = 64 while the count started at 3.7 K).
        lam = lam * math.exp(min((ek.item() - K) / K, 1.0) / B_H)
    if dev == 'cuda':
        torch.cuda.synchronize()
    step_seconds.append(time.time() - t_step)
    if (step + 1) % LOG_EVERY == 0:
        # The loss curve between evaluations: every 0.4M training tokens, the step's training KL and F, its bits per
        # clean token (training gates and hard), and lambda.
        with open(out.replace('.json', '.tsv'), 'a') as f_:
            f_.write(f"{step + 1}\t{(step + 1) * batch * seq}\t{kl.item():.4f}\t{objective.item():.4f}\t{ek.item():.0f}\t{hk:.0f}\t{lam:.4g}"
                     + (f"\t{G_t.item():.0f}\t{kl_dl:.4f}" if CONCEPTS else '') + "\n")
    last = step == steps - 1 or time.time() - t0 > LIMIT
    if (step + 1) % EVAL == 0 or last:
        # This step's parts on per clean training token, per map (at the sample under F), and under F each
        # kind of tensor's posterior width: the median over tensors of RMS(sigma) / RMS(mu).
        train_per_map = [((h * counted).sum() / counted.sum()).item() for h in train_hard]
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
        rec = {'step': step + 1, 'lambda': lam, 'K_t': K, 'train_kl': kl.item(), 'description_bits': desc.item(), 'train_edge_bits': float(edge_bits),
               'train_gate_H': gate_H.item() if GATE_H else None,
               'train_F': objective.item(), 'train_k_soft': ek.item(), 'train_k_hard': hk, 'train_per_map': train_per_map,
               'sigma_rel': sigma_rel, **e,
               'seconds': time.time() - t0}
        log['trace'].append(rec); print(rec, flush=True); save(step + 1)
        json.dump(log, open(out, 'w'), indent=1)
    if last:
        break

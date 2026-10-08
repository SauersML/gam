"""DESTKV's k-slice signal and gated scores (budget_descent.py's dest_k_scores) as fused Triton kernels.

signal: sig[b, h, t, c] = max over u <= t of |Cf[b, h, u, c] (Qr[b, h, t] . K_c(u))|, no gradient;
scores: s[b, h, t, u] = sum_c G[b, h, t, c] Cf[b, h, u, c] (Qr[b, h, t] . K_c(u)) for u <= t, with gradients in Qr, G,
Cf and U; K_c(u) = rope_u(U[h, c]) = U_c cos_u + rot(U_c) sin_u, rot(w) = cat(-w[n:], w[:n]).
Each program loops over slices and key (or query) blocks with one tl.dot (TF32) per slice and tile, building the
keys in registers, so no [T, T] per slice or [T, HD, C] tensor is written."""
import torch
import triton
import triton.language as tl


_FITS = {}
def _launch(kernel, grid_of, args, consts, tiles):
    """kernel[grid_of(BT, BU)](*args, BT, BU, **consts) at the first tile size in `tiles` the GPU's shared memory takes
    (remembered per kernel)."""
    for i in range(_FITS.get(kernel, 0), len(tiles)):
        BT, BU = tiles[i]
        try:
            out = kernel[grid_of(BT, BU)](*args, BT=BT, BU=BU, **consts)
            _FITS[kernel] = i
            return out
        except triton.runtime.errors.OutOfResources:
            continue
    raise RuntimeError('dest_kernels: no tile size fits shared memory')


TILES = ((64, 64), (32, 32), (16, 16))


@triton.jit
def _dest_signal(Q, Cf, U, COS, SIN, OUT, T, C,
                 sq_bh, sq_t, sc_bh, sc_t, su_h, su_c, so_bh, so_t,
                 NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BU: tl.constexpr, CB: tl.constexpr):
    bh = tl.program_id(0); tb = tl.program_id(1); cb = tl.program_id(2)
    h = bh % NH
    t = tb * BT + tl.arange(0, BT)
    d = tl.arange(0, HD)
    half: tl.constexpr = HD // 2
    q = tl.load(Q + bh * sq_bh + t[:, None] * sq_t + d[None, :], mask=t[:, None] < T, other=0.0)          # [BT, HD]
    # rot(w)_d = -w_{d + n} for d < n, w_{d - n} for d >= n
    dr = tl.where(d < half, d + half, d - half)
    sg = tl.where(d < half, -1.0, 1.0)
    t_end = tl.minimum(tb * BT + BT, T)
    for ci in range(CB):
        c = cb * CB + ci
        if c < C:
            u_c = tl.load(U + h * su_h + c * su_c + d)                                                      # [HD]
            r_c = tl.load(U + h * su_h + c * su_c + dr) * sg
            best = tl.zeros([BT], dtype=tl.float32)
            for u0 in range(0, t_end, BU):
                u = u0 + tl.arange(0, BU)
                cs = tl.load(COS + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)          # [BU, HD]
                sn = tl.load(SIN + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
                k = u_c[None, :] * cs + r_c[None, :] * sn                                                  # [BU, HD]
                s = tl.dot(q, tl.trans(k))                                                                 # [BT, BU]
                cu = tl.load(Cf + bh * sc_bh + u * sc_t + c, mask=u < T, other=0.0)                         # [BU]
                s = tl.abs(s * cu[None, :])
                s = tl.where((u[None, :] <= t[:, None]) & (u[None, :] < T), s, 0.0)
                best = tl.maximum(best, tl.max(s, axis=1))
            tl.store(OUT + bh * so_bh + t * so_t + c, best, mask=t < T)


def dest_signal(Qr, c, U, cos, sin, CB=16):
    """Qr [B, H, T, HD], c [B, H, T, C], U [H, C, HD], cos/sin [T, HD] (float32, CUDA): sig [B, H, T, C]."""
    B, H, T_, HD = Qr.shape; C = c.shape[-1]
    Qf = Qr.reshape(B * H, T_, HD).contiguous(); cf = c.reshape(B * H, T_, C).contiguous(); Uc = U.contiguous()
    out = torch.empty(B * H, T_, C, device=Qr.device, dtype=torch.float32)
    _launch(_dest_signal, lambda BT, BU: (B * H, triton.cdiv(T_, BT), triton.cdiv(C, CB)),
            (Qf, cf, Uc, cos.contiguous(), sin.contiguous(), out, T_, C, Qf.stride(0), Qf.stride(1), cf.stride(0), cf.stride(1),
             Uc.stride(0), Uc.stride(1), out.stride(0), out.stride(1)), dict(NH=H, HD=HD, CB=CB, num_stages=1), TILES)
    return out.view(B, H, T_, C)



@triton.jit
def _fwd(Q, GT, CT, U, COS, SIN, S, T, C, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BU: tl.constexpr):
    bh = tl.program_id(0); tb = tl.program_id(1); ub = tl.program_id(2)
    if ub <= tb:
        h = bh % NH
        t = tb * BT + tl.arange(0, BT); u = ub * BU + tl.arange(0, BU); d = tl.arange(0, HD)
        half: tl.constexpr = HD // 2
        dr = tl.where(d < half, d + half, d - half); sg = tl.where(d < half, -1.0, 1.0)
        q = tl.load(Q + bh * T * HD + t[:, None] * HD + d[None, :], mask=t[:, None] < T, other=0.0)
        cs = tl.load(COS + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
        sn = tl.load(SIN + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
        acc = tl.zeros([BT, BU], dtype=tl.float32)
        for c in range(C):
            uc = tl.load(U + (h * C + c) * HD + d); rc = tl.load(U + (h * C + c) * HD + dr) * sg
            k = uc[None, :] * cs + rc[None, :] * sn
            w = tl.dot(q, tl.trans(k))
            gt = tl.load(GT + (bh * C + c) * T + t, mask=t < T, other=0.0)
            cu = tl.load(CT + (bh * C + c) * T + u, mask=u < T, other=0.0)
            acc += w * gt[:, None] * cu[None, :]
        keep = (u[None, :] <= t[:, None]) & (t[:, None] < T) & (u[None, :] < T)
        tl.store(S + bh * T * T + t[:, None] * T + u[None, :], tl.where(keep, acc, 0.0), mask=(t[:, None] < T) & (u[None, :] < T))


@triton.jit
def _bwd_q(Q, GT, CT, U, COS, SIN, DS, DQ, T, C, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BU: tl.constexpr):
    """dQr[t] = sum_u sum_c dS[t, u] G[t, c] Cf[u, c] K_c(u)."""
    bh = tl.program_id(0); tb = tl.program_id(1)
    h = bh % NH
    t = tb * BT + tl.arange(0, BT); d = tl.arange(0, HD)
    half: tl.constexpr = HD // 2
    dr = tl.where(d < half, d + half, d - half); sg = tl.where(d < half, -1.0, 1.0)
    acc = tl.zeros([BT, HD], dtype=tl.float32)
    t_end = tl.minimum(tb * BT + BT, T)
    for u0 in range(0, t_end, BU):
        u = u0 + tl.arange(0, BU)
        cs = tl.load(COS + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
        sn = tl.load(SIN + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
        keep = (u[None, :] <= t[:, None]) & (t[:, None] < T) & (u[None, :] < T)
        ds = tl.load(DS + bh * T * T + t[:, None] * T + u[None, :], mask=keep, other=0.0)
        for c in range(C):
            uc = tl.load(U + (h * C + c) * HD + d); rc = tl.load(U + (h * C + c) * HD + dr) * sg
            k = uc[None, :] * cs + rc[None, :] * sn
            gt = tl.load(GT + (bh * C + c) * T + t, mask=t < T, other=0.0)
            cu = tl.load(CT + (bh * C + c) * T + u, mask=u < T, other=0.0)
            a = ds * gt[:, None] * cu[None, :]
            acc += tl.dot(a, k)
    tl.store(DQ + bh * T * HD + t[:, None] * HD + d[None, :], acc, mask=t[:, None] < T)


@triton.jit
def _bwd_g(Q, CT, U, COS, SIN, DS, DGT, T, C, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BU: tl.constexpr, CB: tl.constexpr):
    """dG[t, c] = sum_u dS[t, u] Cf[u, c] (Qr_t . K_c(u)): key blocks outside, the block's slices inside, so each dS
    tile is read once per program."""
    bh = tl.program_id(0); tb = tl.program_id(1); cb = tl.program_id(2)
    h = bh % NH
    t = tb * BT + tl.arange(0, BT); d = tl.arange(0, HD); cols = tl.arange(0, CB)
    half: tl.constexpr = HD // 2
    dr = tl.where(d < half, d + half, d - half); sg = tl.where(d < half, -1.0, 1.0)
    q = tl.load(Q + bh * T * HD + t[:, None] * HD + d[None, :], mask=t[:, None] < T, other=0.0)
    acc = tl.zeros([BT, CB], dtype=tl.float32)
    t_end = tl.minimum(tb * BT + BT, T)
    for u0 in range(0, t_end, BU):
        u = u0 + tl.arange(0, BU)
        cs = tl.load(COS + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
        sn = tl.load(SIN + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
        keep = (u[None, :] <= t[:, None]) & (t[:, None] < T) & (u[None, :] < T)
        ds = tl.load(DS + bh * T * T + t[:, None] * T + u[None, :], mask=keep, other=0.0)
        for ci in range(CB):
            c = tl.minimum(cb * CB + ci, C - 1)
            uc = tl.load(U + (h * C + c) * HD + d); rc = tl.load(U + (h * C + c) * HD + dr) * sg
            k = uc[None, :] * cs + rc[None, :] * sn
            w = tl.dot(q, tl.trans(k))
            cu = tl.load(CT + (bh * C + c) * T + u, mask=u < T, other=0.0)
            v = tl.sum(ds * w * cu[None, :], axis=1)
            acc += tl.where(cols[None, :] == ci, v[:, None], 0.0)
    cc = cb * CB + cols
    tl.store(DGT + (bh * C + cc[None, :]) * T + t[:, None], acc, mask=(t[:, None] < T) & (cc[None, :] < C))


@triton.jit
def _bwd_cu(Q, GT, CT, U, COS, SIN, DS, DCT, DU, T, C, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BU: tl.constexpr, CB: tl.constexpr):
    """dCf[u, c] = sum_t dS[t, u] G[t, c] (Qr_t . K_c(u)) and dU_c[d] = sum_{t, u} X[t, u] (Qr_td cos_ud +
    sg_{pi(d)} Qr_{t, pi(d)} sin_{u, pi(d)}), X = dS G_c(t) Cf_c(u), pi the half swap: query blocks outside, the
    block's slices inside, so each dS and Qr tile is read once per program."""
    bh = tl.program_id(0); ub = tl.program_id(1); cb = tl.program_id(2)
    h = bh % NH
    u = ub * BU + tl.arange(0, BU); d = tl.arange(0, HD); cols = tl.arange(0, CB)
    half: tl.constexpr = HD // 2
    dr = tl.where(d < half, d + half, d - half); sg = tl.where(d < half, -1.0, 1.0)
    sgp = tl.where(d < half, 1.0, -1.0)                                  # sg at pi(d)
    cs = tl.load(COS + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
    sn = tl.load(SIN + u[:, None] * HD + d[None, :], mask=u[:, None] < T, other=0.0)
    snp = tl.load(SIN + u[:, None] * HD + dr[None, :], mask=u[:, None] < T, other=0.0)
    acc_c = tl.zeros([BU, CB], dtype=tl.float32)
    acc_u = tl.zeros([CB, HD], dtype=tl.float32)
    for t0 in range(ub * BU, T, BT):
        t = t0 + tl.arange(0, BT)
        keep = (u[None, :] <= t[:, None]) & (t[:, None] < T) & (u[None, :] < T)
        ds = tl.load(DS + bh * T * T + t[:, None] * T + u[None, :], mask=keep, other=0.0)
        q = tl.load(Q + bh * T * HD + t[:, None] * HD + d[None, :], mask=t[:, None] < T, other=0.0)
        qp = tl.load(Q + bh * T * HD + t[:, None] * HD + dr[None, :], mask=t[:, None] < T, other=0.0) * sgp[None, :]
        for ci in range(CB):
            c = tl.minimum(cb * CB + ci, C - 1)
            uc = tl.load(U + (h * C + c) * HD + d); rc = tl.load(U + (h * C + c) * HD + dr) * sg
            k = uc[None, :] * cs + rc[None, :] * sn
            w = tl.dot(q, tl.trans(k))
            gt = tl.load(GT + (bh * C + c) * T + t, mask=t < T, other=0.0)
            cu = tl.load(CT + (bh * C + c) * T + u, mask=u < T, other=0.0)
            a = ds * gt[:, None]
            vc = tl.sum(a * w, axis=0)
            acc_c += tl.where(cols[None, :] == ci, vc[:, None], 0.0)
            x = a * cu[None, :]
            vu = tl.sum(q * tl.dot(x, cs) + qp * tl.dot(x, snp), axis=0)
            acc_u += tl.where(cols[:, None] == ci, vu[None, :], 0.0)
    cc = cb * CB + cols
    tl.store(DCT + (bh * C + cc[None, :]) * T + u[:, None], acc_c, mask=(u[:, None] < T) & (cc[None, :] < C))
    tl.atomic_add(DU + (h * C + cc[:, None]) * HD + d[None, :], acc_u, mask=cc[:, None] < C)


def _prep(Qr, G, Cf, U):
    B, H, T_, HD = Qr.shape; C = U.shape[1]
    Qf = Qr.reshape(B * H, T_, HD).contiguous()
    GT = G.reshape(B * H, T_, C).transpose(1, 2).contiguous()
    CT = Cf.reshape(B * H, T_, C).transpose(1, 2).contiguous()
    return B, H, T_, HD, C, Qf, GT, CT, U.contiguous()


class DestScores(torch.autograd.Function):
    CB = 16

    @staticmethod
    def forward(ctx, Qr, G, Cf, U, cos, sin):
        B, H, T_, HD, C, Qf, GT, CT, Uc = _prep(Qr, G, Cf, U)
        S = torch.zeros(B * H, T_, T_, device=Qr.device, dtype=torch.float32)
        _launch(_fwd, lambda BT, BU: (B * H, triton.cdiv(T_, BT), triton.cdiv(T_, BU)), (Qf, GT, CT, Uc, cos, sin, S, T_, C),
                dict(NH=H, HD=HD, num_stages=1), TILES)
        ctx.save_for_backward(Qf, GT, CT, Uc, cos, sin)
        ctx.shape = (B, H, T_, HD, C)
        return S.view(B, H, T_, T_)

    @staticmethod
    def backward(ctx, dS):
        Qf, GT, CT, Uc, cos, sin = ctx.saved_tensors
        B, H, T_, HD, C = ctx.shape
        dS = dS.reshape(B * H, T_, T_).contiguous()
        CB = DestScores.CB
        dQ = torch.empty_like(Qf); dGT = torch.empty_like(GT); dCT = torch.empty_like(CT); dU = torch.zeros_like(Uc)
        _launch(_bwd_q, lambda BT, BU: (B * H, triton.cdiv(T_, BT)), (Qf, GT, CT, Uc, cos, sin, dS, dQ, T_, C),
                dict(NH=H, HD=HD, num_stages=1), TILES)
        _launch(_bwd_g, lambda BT, BU: (B * H, triton.cdiv(T_, BT), triton.cdiv(C, CB)), (Qf, CT, Uc, cos, sin, dS, dGT, T_, C),
                dict(NH=H, HD=HD, CB=CB, num_warps=8, num_stages=1), TILES)
        _launch(_bwd_cu, lambda BT, BU: (B * H, triton.cdiv(T_, BU), triton.cdiv(C, CB)), (Qf, GT, CT, Uc, cos, sin, dS, dCT, dU, T_, C),
                dict(NH=H, HD=HD, CB=CB, num_warps=8, num_stages=1), TILES)
        return (dQ.view(B, H, T_, HD), dGT.transpose(1, 2).reshape(B, H, T_, C), dCT.transpose(1, 2).reshape(B, H, T_, C), dU, None, None)

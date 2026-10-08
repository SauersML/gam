"""SENSGATE's draws and per-slice sums of squared products (budget_descent.py's sens_pass) as fused Triton kernels.

sample: y[n, k] = the first index at which the cumulative sum of softmax(L[n]) reaches u[n, k], from two reads of each
row of logits; no float32 [N, V] tensor is written.
draw_sq: out[b, m, c] (+)= sum over k of (sum_j A[k, b, m, j] B[b, c, j])^2, the K products and their squares summed
in registers, so no [K, M, C] tensor is written; bfloat16 operands, float32 sums and output. With `out` given the
sums add into it (one read and one write of it per call)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _sample(L, Ud, Y, V, sl, K: tl.constexpr, KP: tl.constexpr, BV: tl.constexpr):
    n = tl.program_id(0)
    v = tl.arange(0, BV)
    # Pass 1: the row's max and normalizer, per lane, then across lanes.
    m = tl.full([BV], float('-inf'), tl.float32)
    s = tl.zeros([BV], tl.float32)
    for v0 in range(0, V, BV):
        x = tl.load(L + n * sl + v0 + v, mask=v0 + v < V, other=float('-inf')).to(tl.float32)
        m_new = tl.maximum(m, x)
        s = s * tl.where(m == float('-inf'), 0.0, tl.exp(m - m_new)) + tl.where(x == float('-inf'), 0.0, tl.exp(x - m_new))
        m = m_new
    M = tl.max(m, axis=0)
    S = tl.sum(s * tl.where(m == float('-inf'), 0.0, tl.exp(m - M)), axis=0)
    # Pass 2: p's running sum, and per draw the count of indices whose running sum is below u (the sampled index).
    k = tl.arange(0, KP)
    u = tl.load(Ud + n * K + k, mask=k < K, other=2.0)
    cnt = tl.zeros([KP], tl.int32)
    run = tl.sum(tl.zeros([BV], tl.float32), axis=0)
    for v0 in range(0, V, BV):
        x = tl.load(L + n * sl + v0 + v, mask=v0 + v < V, other=float('-inf')).to(tl.float32)
        p = tl.exp(x - M) / S
        cs = tl.cumsum(p, 0) + run
        cnt += tl.sum(((cs[:, None] < u[None, :]) & ((v0 + v)[:, None] < V)).to(tl.int32), axis=0)
        run += tl.sum(p, axis=0)
    tl.store(Y + n * K + k, tl.minimum(cnt, V - 1).to(tl.int64), mask=k < K)


def sample(logits, K):
    """K draws per row of softmax(logits) [N, V] (-inf logits never drawn) -> y [N, K] int64."""
    N, V = logits.shape
    assert logits.stride(1) == 1
    u = torch.rand(N, K, device=logits.device)
    y = torch.empty(N, K, device=logits.device, dtype=torch.int64)
    _sample[(N,)](logits, u, y, V, logits.stride(0), K=K, KP=max(2, triton.next_power_of_2(K)), BV=2048, num_warps=8)
    return y


@triton.jit
def _draw_sq(A, B, O, M, C, J, sa_k, sa_b, sa_m, sb_b, sb_c, so_b, so_m,
             K: tl.constexpr, ACC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BJ: tl.constexpr):
    # c blocks vary fastest, so the programs running together share their A tiles (the K cotangents' rows) in L2.
    cb = tl.program_id(0); mb = tl.program_id(1); b = tl.program_id(2)
    m = mb * BM + tl.arange(0, BM)
    c = cb * BN + tl.arange(0, BN)
    j = tl.arange(0, BJ)
    msk = (m[:, None] < M) & (c[None, :] < C)
    po = O + b * so_b + m[:, None] * so_m + c[None, :]
    if ACC:
        sq = tl.load(po, mask=msk, other=0.0)
    else:
        sq = tl.zeros([BM, BN], tl.float32)
    for k in range(K):
        acc = tl.zeros([BM, BN], tl.float32)
        for j0 in range(0, J, BJ):
            a = tl.load(A + k * sa_k + b * sa_b + m[:, None] * sa_m + (j0 + j)[None, :], mask=(m[:, None] < M) & ((j0 + j)[None, :] < J), other=0.0)
            w = tl.load(B + b * sb_b + c[:, None] * sb_c + (j0 + j)[None, :], mask=(c[:, None] < C) & ((j0 + j)[None, :] < J), other=0.0)
            acc += tl.dot(a, tl.trans(w))
        sq += acc * acc
    tl.store(po, sq, mask=msk)


def draw_sq(A, B, nb, M, J, sa, out=None):
    """sum_k (A_k B^T)^2 per b, as [nb, M, C] float32 (added into `out` when given): A's element (k, b, m, j) at
    A + k sa[0] + b sa[1] + m sa[2] + j, B [nb, C, J] (or [C, J] when nb is 1) with unit stride in j."""
    B = B.reshape(nb, *B.shape[-2:])
    assert A.dtype == B.dtype == torch.bfloat16 and B.stride(2) == 1 and A.stride(-1) == 1
    C = B.shape[1]
    acc = out is not None
    if out is None:
        out = torch.empty(nb, M, C, device=A.device, dtype=torch.float32)
    assert out.dtype == torch.float32 and out.stride(2) == 1
    grid = (triton.cdiv(C, 128), triton.cdiv(M, 128), nb)
    _draw_sq[grid](A, B, out, M, C, J, sa[0], sa[1], sa[2], B.stride(0), B.stride(1), out.stride(0), out.stride(1),
                   K=A.shape[0], ACC=acc, BM=128, BN=128, BJ=32, num_warps=8, num_stages=3)
    return out

"""
CTN pairwise psi-psi second-order ground truth.

Mirrors the Rust toy fixture in
  src/families/transformation_normal.rs::tests::toy_covariate_design_and_derivs

and produces, at fixed beta (NO inner Newton refinement), the three pairwise
quantities that the Rust pairwise body returns as
`ExactNewtonJointPsiSecondOrderTerms`:

    pair_a    [a][b]   scalar    = d^2 (-log L) / d psi_a d psi_b
    pair_g    [a][b]   p-vector  = d^2 (grad_beta -log L) / d psi_a d psi_b
    pair_b_mat[a][b]   p x p     = d^2 (Hess_beta -log L) / d psi_a d psi_b

where -log L = sum_i w_i [ 1/2 h_i^2 - log h'_i ] is the CTN negative
log-likelihood (no penalty term, no logdet pieces -- this is the
likelihood-only block the Rust pairwise body computes).

We also compute the directional contractions

    a_dir[i] = sum_j v[j] * pair_a[i][j]
    g_dir[i] = sum_j v[j] * pair_g[i][j]
    b_dir[i] = sum_j v[j] * pair_b_mat[i][j]

Method: closed-form analytic differentiation. C(x; psi) is quadratic in psi,
so its second psi-derivatives are constants and third derivatives are zero.
The likelihood in beta is built from h(beta, psi) = R . C(psi) . beta and
h'(beta, psi) = R' . C(psi) . beta (rank-1 tensor); we differentiate this
exactly. Beta is held fixed throughout (no implicit beta*(psi) chain).

Indexing convention: beta is a length-p vector with p = p_resp * p_cov, and
beta[j*p_cov + k] is the coefficient for response basis component j and
covariate basis component k (matches Rust Khatri-Rao layout).
"""

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Toy fixture (mirrors transformation_normal.rs::toy_covariate_design_and_derivs)
# ---------------------------------------------------------------------------

x0  = np.array([[1.00, 0.40], [1.10, 0.35], [1.20, 0.45], [0.95, 0.50]])
x_a = np.array([[0.10, -0.02], [0.08, 0.01], [0.12, -0.01], [0.09, 0.03]])
x_b = np.array([[-0.04, 0.06], [-0.02, 0.05], [-0.03, 0.04], [-0.01, 0.07]])
x_aa = np.array([[0.02, 0.00], [0.01, 0.01], [0.02, -0.01], [0.01, 0.02]])
x_ab = np.array([[0.01, -0.01], [0.00, 0.02], [0.01, 0.01], [0.00, -0.01]])
x_bb = np.array([[-0.01, 0.02], [-0.02, 0.01], [-0.01, 0.00], [-0.02, 0.02]])

response_val_basis   = np.array([[1.0, -1.0], [1.0, -0.2], [1.0, 0.6], [1.0, 1.3]])
response_deriv_basis = np.array([[0.0, 1.0],  [0.0, 1.0],  [0.0, 1.0], [0.0, 1.0]])
weights = np.array([1.0, 1.0, 1.0, 1.0])

n = response_val_basis.shape[0]      # 4
p_resp = response_val_basis.shape[1] # 2
p_cov  = x0.shape[1]                 # 2
p = p_resp * p_cov                   # 4

psi = np.array([0.15, -0.10])
psi_dim = 2
beta = np.array([0.15, -0.05, 0.80, 0.30])  # beta[j*p_cov + k]
v = np.array([0.4, -0.7])

# ---------------------------------------------------------------------------
# 2. C(psi) and its psi-derivatives at this psi.
# C is shape (n, p_cov); C_psi[m] is shape (n, p_cov); C_psi2[m1][m2] is constant.
# Quadratic in psi:
#   C        = x0 + psi0 x_a + psi1 x_b + 0.5 psi0^2 x_aa + psi0 psi1 x_ab + 0.5 psi1^2 x_bb
#   dC/dpsi0 = x_a + psi0 x_aa + psi1 x_ab
#   dC/dpsi1 = x_b + psi0 x_ab + psi1 x_bb
#   d2C/dpsi0^2     = x_aa
#   d2C/dpsi0 dpsi1 = x_ab
#   d2C/dpsi1^2     = x_bb
# Third and higher psi-derivatives are zero.
# ---------------------------------------------------------------------------
C        = (x0 + psi[0]*x_a + psi[1]*x_b
            + 0.5*psi[0]**2 * x_aa + psi[0]*psi[1] * x_ab + 0.5*psi[1]**2 * x_bb)
C_psi    = [x_a + psi[0]*x_aa + psi[1]*x_ab,
            x_b + psi[0]*x_ab + psi[1]*x_bb]
C_psi2   = [[x_aa, x_ab],
            [x_ab, x_bb]]

# ---------------------------------------------------------------------------
# 3. Per-observation Khatri-Rao tensor pieces.
# For each row i, define the (p_resp*p_cov)-vector
#    phi_i        [j*p_cov + k]   = R_val[i,j]   * C[i,k]
#    phi_prime_i  [j*p_cov + k]   = R_deriv[i,j] * C[i,k]
# So h_i  = phi_i . beta  and  h'_i = phi_prime_i . beta.
#
# Psi-derivatives propagate only through C:
#    d phi_i / d psi_m         [j,k] = R_val[i,j]   * C_psi[m][i,k]
#    d^2 phi_i / d psi_m d psi_n[j,k] = R_val[i,j]   * C_psi2[m][n][i,k]
# (and analogously for phi_prime via R_deriv).
# ---------------------------------------------------------------------------

def kron_per_row(R_row, C_row):
    r"""Return the length-p Khatri-Rao row R_row \otimes C_row (Kronecker)."""
    return np.outer(R_row, C_row).reshape(-1)

phi       = np.array([kron_per_row(response_val_basis[i],   C[i])    for i in range(n)])
phi_prime = np.array([kron_per_row(response_deriv_basis[i], C[i])    for i in range(n)])

# psi first derivatives of phi, phi'.
phi_psi  = [np.array([kron_per_row(response_val_basis[i],   C_psi[m][i])  for i in range(n)])
            for m in range(psi_dim)]
phip_psi = [np.array([kron_per_row(response_deriv_basis[i], C_psi[m][i])  for i in range(n)])
            for m in range(psi_dim)]

# psi second derivatives (constants in psi).
phi_psi2  = [[np.array([kron_per_row(response_val_basis[i],   C_psi2[m1][m2][i])  for i in range(n)])
              for m2 in range(psi_dim)] for m1 in range(psi_dim)]
phip_psi2 = [[np.array([kron_per_row(response_deriv_basis[i], C_psi2[m1][m2][i])  for i in range(n)])
              for m2 in range(psi_dim)] for m1 in range(psi_dim)]

# ---------------------------------------------------------------------------
# 4. Per-observation likelihood scalars and their psi-derivatives.
#
# Define for each i:
#   h_i  = phi_i        . beta
#   hp_i = phi_prime_i  . beta
#
# Negative log-likelihood per row (no penalty, no logdet):
#   L_i = w_i * (1/2 h_i^2 - log hp_i)
#
# Total negative log-likelihood:
#   F_lik(beta, psi) = sum_i L_i
#
# Beta-gradient of L_i:
#   dL_i/dbeta = w_i * (h_i * phi_i - phi_prime_i / hp_i)
#
# Beta-Hessian of L_i:
#   d^2 L_i / dbeta dbeta^T
#       = w_i * ( phi_i phi_i^T  +  phi_prime_i phi_prime_i^T / hp_i^2 )
#
# We need second psi-derivatives at fixed beta.
# Let q_m = d/dpsi_m, q_mn = d^2/dpsi_m dpsi_n.
#   q_m h_i  = (q_m phi_i)  . beta
#   q_m hp_i = (q_m phi'_i) . beta
#   q_mn h_i  = (q_mn phi_i)  . beta
#   q_mn hp_i = (q_mn phi'_i) . beta
# (these are the only psi-derivatives we need; higher orders not used.)
# ---------------------------------------------------------------------------

h    = phi        @ beta              # (n,)
hp   = phi_prime  @ beta              # (n,)
h_m  = [phi_psi[m]  @ beta for m in range(psi_dim)]
hp_m = [phip_psi[m] @ beta for m in range(psi_dim)]
h_mn  = [[phi_psi2[m1][m2]  @ beta for m2 in range(psi_dim)] for m1 in range(psi_dim)]
hp_mn = [[phip_psi2[m1][m2] @ beta for m2 in range(psi_dim)] for m1 in range(psi_dim)]

assert np.all(hp > 0.0), f"hp must be positive at this fixture; got {hp}"

# Quick sanity: log L value and its psi gradient.
F_lik = float(np.sum(weights * (0.5 * h**2 - np.log(hp))))
log_L = float(np.sum(weights * (-0.5 * h**2 + np.log(hp))))

# ---------------------------------------------------------------------------
# 5. pair_a[a][b] = d^2 F_lik / d psi_a d psi_b at fixed beta.
#
# Per row i, F_lik_i = w_i ( 1/2 h_i^2 - log hp_i ).
#  d/dpsi_m F_lik_i           = w_i ( h_i  q_m h_i  -  q_m hp_i / hp_i )
#  d^2/dpsi_m dpsi_n F_lik_i  = w_i ( q_n h_i * q_m h_i + h_i * q_mn h_i
#                                     -  q_mn hp_i / hp_i
#                                     +  q_m hp_i * q_n hp_i / hp_i^2 )
# Here we use the identity d/dpsi_n (1/hp) = -q_n hp / hp^2.
# ---------------------------------------------------------------------------
pair_a = np.zeros((psi_dim, psi_dim))
for a in range(psi_dim):
    for b in range(psi_dim):
        per_row = weights * (
            h_m[b] * h_m[a]
            + h * h_mn[a][b]
            - hp_mn[a][b] / hp
            + (hp_m[a] * hp_m[b]) / (hp * hp)
        )
        pair_a[a, b] = per_row.sum()

# ---------------------------------------------------------------------------
# 6. pair_g[a][b] = d^2 (dF_lik/dbeta) / d psi_a d psi_b at fixed beta.
#
# dF_lik/dbeta_per_row = w_i ( h_i * phi_i - phi'_i / hp_i ).
#
# Differentiate twice in psi (note phi_i, phi'_i depend on psi; h_i, hp_i too).
#
# Let A_i  := h_i * phi_i.       Then
#   q_m A_i = (q_m h_i) phi_i + h_i (q_m phi_i)
#   q_mn A_i = (q_mn h_i) phi_i + (q_m h_i)(q_n phi_i) + (q_n h_i)(q_m phi_i) + h_i (q_mn phi_i)
#
# Let B_i := phi'_i / hp_i.      Then with u := 1/hp_i:
#   q_m u = -q_m hp_i / hp_i^2
#   q_mn u = -q_mn hp_i/hp_i^2 + 2 q_m hp_i q_n hp_i / hp_i^3
#   q_m B_i  = (q_m phi'_i) u + phi'_i (q_m u)
#   q_mn B_i = (q_mn phi'_i) u + (q_m phi'_i)(q_n u) + (q_n phi'_i)(q_m u) + phi'_i (q_mn u)
#
# pair_g[a][b] = sum_i w_i ( q_ab A_i - q_ab B_i )
# ---------------------------------------------------------------------------
pair_g = np.zeros((psi_dim, psi_dim, p))

inv_hp  = 1.0 / hp
inv_hp2 = inv_hp * inv_hp
inv_hp3 = inv_hp2 * inv_hp

for a in range(psi_dim):
    for b in range(psi_dim):
        # q_ab A_i, summed over i with weight w_i.
        # Each term in the per-row sum is a length-p vector.
        # Build the row-wise outer product on phi quantities:  scalar_i * vec_i[k].
        term_A = (
            h_mn[a][b][:, None] * phi              # (q_ab h_i) * phi_i
            + h_m[a][:, None] * phi_psi[b]         # (q_a h_i) * (q_b phi_i)
            + h_m[b][:, None] * phi_psi[a]         # (q_b h_i) * (q_a phi_i)
            + h[:, None] * phi_psi2[a][b]          # h_i * (q_ab phi_i)
        )
        # u = 1/hp; its derivatives:
        u    = inv_hp
        u_a  = -hp_m[a] * inv_hp2
        u_b  = -hp_m[b] * inv_hp2
        u_ab = -hp_mn[a][b] * inv_hp2 + 2.0 * hp_m[a] * hp_m[b] * inv_hp3
        term_B = (
            phip_psi2[a][b] * u[:, None]           # (q_ab phi'_i) * u
            + phip_psi[a]  * u_b[:, None]          # (q_a phi'_i) * (q_b u)
            + phip_psi[b]  * u_a[:, None]          # (q_b phi'_i) * (q_a u)
            + phi_prime    * u_ab[:, None]         # phi'_i * (q_ab u)
        )
        per_row = weights[:, None] * (term_A - term_B)   # (n, p)
        pair_g[a, b] = per_row.sum(axis=0)

# ---------------------------------------------------------------------------
# 7. pair_b_mat[a][b] = d^2 (d^2 F_lik/dbeta dbeta^T) / d psi_a d psi_b.
#
# Per-row beta-Hessian:
#   M_i = w_i ( phi_i phi_i^T  +  phi'_i phi'_i^T / hp_i^2 )
#
# Let P_i := phi_i phi_i^T.
#   q_m P_i = (q_m phi_i) phi_i^T + phi_i (q_m phi_i)^T
#   q_mn P_i = (q_mn phi_i) phi_i^T + (q_m phi_i)(q_n phi_i)^T
#            + (q_n phi_i)(q_m phi_i)^T + phi_i (q_mn phi_i)^T
#
# Let Q_i := phi'_i phi'_i^T / hp_i^2 = phi'_i phi'_i^T * v   with  v := 1/hp_i^2.
#   q_m v  = -2 q_m hp_i / hp_i^3
#   q_mn v = -2 q_mn hp_i/hp_i^3 + 6 q_m hp_i q_n hp_i / hp_i^4
# Then with R_i := phi'_i phi'_i^T:
#   q_m R_i  = (q_m phi'_i) phi'_i^T + phi'_i (q_m phi'_i)^T
#   q_mn R_i = (q_mn phi'_i) phi'_i^T + (q_m phi'_i)(q_n phi'_i)^T
#             + (q_n phi'_i)(q_m phi'_i)^T + phi'_i (q_mn phi'_i)^T
#   q_m Q_i  = (q_m R_i) v + R_i (q_m v)
#   q_mn Q_i = (q_mn R_i) v + (q_m R_i)(q_n v) + (q_n R_i)(q_m v) + R_i (q_mn v)
#
# pair_b_mat[a][b] = sum_i w_i ( q_ab P_i + q_ab Q_i )
# ---------------------------------------------------------------------------
def per_row_outer(u_mat, vmat):
    """Per-row outer u * v^T (no symmetrization), shape (n, p, p)."""
    return np.einsum('ij,ik->ijk', u_mat, vmat)

pair_b_mat = np.zeros((psi_dim, psi_dim, p, p))

# Cache rank-1 outers we reuse.
P_outer       = per_row_outer(phi, phi)                        # phi phi^T
Pp_outer      = [per_row_outer(phi_psi[m], phi) for m in range(psi_dim)]      # (q_m phi) phi^T
R_outer       = per_row_outer(phi_prime, phi_prime)            # phi' phi'^T
Rp_outer      = [per_row_outer(phip_psi[m], phi_prime) for m in range(psi_dim)]  # (q_m phi') phi'^T

v_scalar    = inv_hp2
v_a   = [-2.0 * hp_m[m] * inv_hp3 for m in range(psi_dim)]
# v_ab depends on (a,b); compute inside the loop.

for a in range(psi_dim):
    for b in range(psi_dim):
        # q_ab P_i (per-row p x p):
        qab_P = (
            per_row_outer(phi_psi2[a][b], phi)
            + per_row_outer(phi_psi[a], phi_psi[b])
            + per_row_outer(phi_psi[b], phi_psi[a])
            + per_row_outer(phi, phi_psi2[a][b])
        )

        # q_ab R_i (per-row p x p):
        qab_R = (
            per_row_outer(phip_psi2[a][b], phi_prime)
            + per_row_outer(phip_psi[a], phip_psi[b])
            + per_row_outer(phip_psi[b], phip_psi[a])
            + per_row_outer(phi_prime, phip_psi2[a][b])
        )
        # q_a R_i and q_b R_i for the cross terms
        qa_R = Rp_outer[a] + per_row_outer(phi_prime, phip_psi[a])
        qb_R = Rp_outer[b] + per_row_outer(phi_prime, phip_psi[b])

        # v derivatives.
        v_b_m = -2.0 * hp_m[b] * inv_hp3
        v_a_m = -2.0 * hp_m[a] * inv_hp3
        v_ab = -2.0 * hp_mn[a][b] * inv_hp3 + 6.0 * hp_m[a] * hp_m[b] * inv_hp2 * inv_hp2

        # q_ab Q_i = (q_ab R) v + (q_a R)(q_b v) + (q_b R)(q_a v) + R (q_ab v)
        qab_Q = (
            qab_R * v_scalar[:, None, None]
            + qa_R * v_b_m[:, None, None]
            + qb_R * v_a_m[:, None, None]
            + R_outer * v_ab[:, None, None]
        )

        per_row_M = weights[:, None, None] * (qab_P + qab_Q)
        pair_b_mat[a, b] = per_row_M.sum(axis=0)

# ---------------------------------------------------------------------------
# 8. Symmetry checks (a,b) <-> (b,a) — must hold by Clairaut.
# ---------------------------------------------------------------------------
asym_a = np.max(np.abs(pair_a - pair_a.T))
asym_g = np.max(np.abs(pair_g - pair_g.transpose(1, 0, 2)))
asym_b = np.max(np.abs(pair_b_mat - pair_b_mat.transpose(1, 0, 2, 3)))
# Within each pair, b_mat[a][b] is also a symmetric p x p matrix (Hessian).
sym_b_inner = np.max([np.max(np.abs(pair_b_mat[a, b] - pair_b_mat[a, b].T))
                      for a in range(psi_dim) for b in range(psi_dim)])

print(f"[sym] pair_a (a,b) <-> (b,a)   max asym = {asym_a:.3e}")
print(f"[sym] pair_g (a,b) <-> (b,a)   max asym = {asym_g:.3e}")
print(f"[sym] pair_b (a,b) <-> (b,a)   max asym = {asym_b:.3e}")
print(f"[sym] pair_b inner symmetry    max asym = {sym_b_inner:.3e}")

# ---------------------------------------------------------------------------
# 9. Directional contractions with v.
# ---------------------------------------------------------------------------
a_dir = np.einsum('j,ij->i', v, pair_a)
g_dir = np.einsum('j,ijk->ik', v, pair_g)
b_dir = np.einsum('j,ijkl->ikl', v, pair_b_mat)

# ---------------------------------------------------------------------------
# 10. Aux diagnostics.
# ---------------------------------------------------------------------------
H_lik = np.zeros((p, p))                       # beta-Hessian of -log L at (beta, psi)
g_lik = np.zeros(p)                            # beta-gradient of -log L at (beta, psi)
for i in range(n):
    g_lik += weights[i] * (h[i] * phi[i] - phi_prime[i] / hp[i])
    H_lik += weights[i] * (np.outer(phi[i], phi[i])
                           + np.outer(phi_prime[i], phi_prime[i]) / (hp[i]**2))
sign_H, H_logdet = np.linalg.slogdet(H_lik)
H_eigs = np.linalg.eigvalsh(0.5 * (H_lik + H_lik.T)).tolist()

# ---------------------------------------------------------------------------
# 11. Cross-check a, g, b via centered FD on the analytic per-row pieces.
# We FD F_lik (and grad_beta, Hess_beta) wrt psi to ensure analytic matches.
# ---------------------------------------------------------------------------
def lik_pieces(psi_local, beta_local):
    C_loc = (x0 + psi_local[0]*x_a + psi_local[1]*x_b
             + 0.5*psi_local[0]**2 * x_aa + psi_local[0]*psi_local[1] * x_ab
             + 0.5*psi_local[1]**2 * x_bb)
    phi_loc       = np.array([kron_per_row(response_val_basis[i],   C_loc[i]) for i in range(n)])
    phi_prime_loc = np.array([kron_per_row(response_deriv_basis[i], C_loc[i]) for i in range(n)])
    h_loc  = phi_loc       @ beta_local
    hp_loc = phi_prime_loc @ beta_local
    F = float(np.sum(weights * (0.5 * h_loc**2 - np.log(hp_loc))))
    g = np.zeros(p)
    H = np.zeros((p, p))
    for i in range(n):
        g += weights[i] * (h_loc[i] * phi_loc[i] - phi_prime_loc[i] / hp_loc[i])
        H += weights[i] * (np.outer(phi_loc[i], phi_loc[i])
                           + np.outer(phi_prime_loc[i], phi_prime_loc[i]) / (hp_loc[i]**2))
    return F, g, H

def fd_pair(a, b, h_step=1e-4):
    """Centered FD for d^2/d psi_a d psi_b of (F, g, H). Uses the 2D 4-point
    centered cross stencil for off-diagonal and centered second-difference
    for diagonal."""
    if a == b:
        e = np.zeros(psi_dim); e[a] = h_step
        Fpp, gpp, Hpp = lik_pieces(psi + e, beta)
        Fmm, gmm, Hmm = lik_pieces(psi - e, beta)
        Fz, gz, Hz = lik_pieces(psi, beta)
        return ((Fpp - 2*Fz + Fmm) / h_step**2,
                (gpp - 2*gz + gmm) / h_step**2,
                (Hpp - 2*Hz + Hmm) / h_step**2)
    ea = np.zeros(psi_dim); ea[a] = h_step
    eb = np.zeros(psi_dim); eb[b] = h_step
    Fpp, gpp, Hpp = lik_pieces(psi + ea + eb, beta)
    Fpm, gpm, Hpm = lik_pieces(psi + ea - eb, beta)
    Fmp, gmp, Hmp = lik_pieces(psi - ea + eb, beta)
    Fmm, gmm, Hmm = lik_pieces(psi - ea - eb, beta)
    return ((Fpp - Fpm - Fmp + Fmm) / (4 * h_step**2),
            (gpp - gpm - gmp + gmm) / (4 * h_step**2),
            (Hpp - Hpm - Hmp + Hmm) / (4 * h_step**2))

# Run FD comparison at h = 1e-4 (Richardson noise budget allows ~ h^2 ~ 1e-8).
fd_a = np.zeros_like(pair_a)
fd_g = np.zeros_like(pair_g)
fd_b = np.zeros_like(pair_b_mat)
for a in range(psi_dim):
    for b in range(psi_dim):
        fd_a[a, b], fd_g[a, b], fd_b[a, b] = fd_pair(a, b, h_step=1e-4)

err_a = float(np.max(np.abs(fd_a - pair_a)))
err_g = float(np.max(np.abs(fd_g - pair_g)))
err_b = float(np.max(np.abs(fd_b - pair_b_mat)))
print(f"[FD-vs-analytic] pair_a max abs diff = {err_a:.3e}")
print(f"[FD-vs-analytic] pair_g max abs diff = {err_g:.3e}")
print(f"[FD-vs-analytic] pair_b max abs diff = {err_b:.3e}")

# ---------------------------------------------------------------------------
# 12. Write JSON.
# ---------------------------------------------------------------------------
out = {
    "config": {
        "n": n,
        "p_resp": p_resp,
        "p_cov": p_cov,
        "p": p,
        "psi_dim": psi_dim,
        "x0": x0.tolist(),
        "x_a": x_a.tolist(),
        "x_b": x_b.tolist(),
        "x_aa": x_aa.tolist(),
        "x_ab": x_ab.tolist(),
        "x_bb": x_bb.tolist(),
        "response_val_basis":   response_val_basis.tolist(),
        "response_deriv_basis": response_deriv_basis.tolist(),
        "weights": weights.tolist(),
        "psi": psi.tolist(),
        "beta": beta.tolist(),
        "v": v.tolist(),
        "beta_index_layout": "beta[j*p_cov + k] = coef(response j, covariate k)",
    },
    "ground_truth": {
        "pair_a":     pair_a.tolist(),                  # (psi_dim, psi_dim)
        "pair_g":     pair_g.tolist(),                  # (psi_dim, psi_dim, p)
        "pair_b_mat": pair_b_mat.tolist(),              # (psi_dim, psi_dim, p, p)
        "a_dir":      a_dir.tolist(),                   # (psi_dim,)
        "g_dir":      g_dir.tolist(),                   # (psi_dim, p)
        "b_dir":      b_dir.tolist(),                   # (psi_dim, p, p)
    },
    "intermediate_quantities": {
        "F_lik_at_(beta,psi)": F_lik,
        "log_L":               log_L,
        "h_per_row":            h.tolist(),
        "hp_per_row":           hp.tolist(),
        "grad_beta_neglogL":    g_lik.tolist(),
        "Hess_beta_neglogL":    H_lik.tolist(),
        "Hess_beta_logdet":     float(H_logdet),
        "Hess_beta_eigvals":    H_eigs,
    },
    "diagnostics": {
        "pair_a_FD_vs_analytic_max_abs": err_a,
        "pair_g_FD_vs_analytic_max_abs": err_g,
        "pair_b_FD_vs_analytic_max_abs": err_b,
        "pair_a_swap_asym_max":          float(asym_a),
        "pair_g_swap_asym_max":          float(asym_g),
        "pair_b_swap_asym_max":          float(asym_b),
        "pair_b_inner_sym_max":          float(sym_b_inner),
        "FD_step_used":                  1e-4,
    },
}

# Primary deliverable per teammate instruction: scripts/ctn_pairwise_groundtruth.json
out_path = Path("/Users/user/gam/scripts/ctn_pairwise_groundtruth.json")
out_path.write_text(json.dumps(out, indent=2))
# Convenience copy at /tmp for direct diff against the Rust oracle output.
tmp_path = Path("/tmp/ctn_pairwise_groundtruth.json")
tmp_path.write_text(json.dumps(out, indent=2))
print(f"[write] {out_path}")
print(f"[write] {tmp_path}")

# ---------------------------------------------------------------------------
# 13. Console summary.
# ---------------------------------------------------------------------------
print("\n=== TOY CONFIG ===")
print(f"n={n} p_resp={p_resp} p_cov={p_cov} p={p} psi_dim={psi_dim}")
print(f"psi  = {psi.tolist()}")
print(f"beta = {beta.tolist()}")
print(f"v    = {v.tolist()}")
print(f"F_lik(beta,psi) = {F_lik:.10f}    log L = {log_L:.10f}")
print(f"h  = {h.tolist()}")
print(f"hp = {hp.tolist()}")

print("\n=== pair_a (psi-psi 2nd derivative of -log L, scalar) ===")
print(pair_a)

print("\n=== pair_g (psi-psi 2nd derivative of grad_beta -log L, p-vec per pair) ===")
for a in range(psi_dim):
    for b in range(psi_dim):
        print(f"pair_g[{a}][{b}] = {pair_g[a, b].tolist()}")

print("\n=== pair_b_mat (psi-psi 2nd derivative of Hess_beta -log L, p x p per pair) ===")
for a in range(psi_dim):
    for b in range(psi_dim):
        print(f"pair_b_mat[{a}][{b}] =\n{pair_b_mat[a, b]}")

print("\n=== Directional contractions with v ===")
print(f"a_dir = {a_dir.tolist()}")
print(f"g_dir =\n{g_dir}")
print(f"b_dir[0] =\n{b_dir[0]}")
print(f"b_dir[1] =\n{b_dir[1]}")

"""
Sympy ground-truth derivation of the CTN outer LAML Hessian-vector product.

Setup (mirrors the teammate brief):

  Parameters:    theta = (rho, psi_0, psi_1)        (q = 3)
  Coefficients:  beta in R^4 (Khatri-Rao tensor: 2 response x 2 covariate)
  Response basis:    R_0(y) = 1,    R_1(y) = y,    R'_0 = 0,  R'_1 = 1
  Covariate basis:   C_0(x; psi) = exp(-psi_0 (x - 0.0)^2)
                     C_1(x; psi) = exp(-psi_1 (x - 1.0)^2)
  Khatri-Rao value:  h(y, beta, psi)  = sum_{a,b} R_a(y)  C_b(x; psi) beta_{a,b}
  Khatri-Rao deriv:  h'(y, beta, psi) = sum_{a,b} R'_a(y) C_b(x; psi) beta_{a,b}

  Penalized neg-log-lik:
      F(beta, theta) = - sum_i w_i [ log h'_i - 1/2 h_i^2 ] + 1/2 beta^T S(rho) beta
  with S(rho) = exp(rho) * I_4.

  Outer LAML:
      V(theta) = F(beta*(theta), theta) + 1/2 log|H| - 1/2 log_pseudo|S|
  where H = F_{beta beta}, beta*(theta) = argmin_beta F.

  Outer gradient and Hessian (with implicit differentiation
  H beta_v = -F_{beta theta}[v]):

      g_i        = F_{theta_i}                 + 1/2 tr(H^{-1} H_{theta_i})
                                                  - 1/2 tr(S^+ S_{theta_i})
      (H_V v)_i  = F_{theta_i theta}[v]
                 + F_{theta_i beta}[beta_v]
                 + 1/2 d_v tr(H^{-1} H_{theta_i})
                 - 1/2 d_v tr(S^+ S_{theta_i})

  CTN structural facts used here:
    * rho only enters S, so F_{rho beta} = S beta and H_{rho beta} = 0.
    * psi only enters C(x; psi); F_{psi *} are nonzero through h, h'.

We derive everything symbolically with sympy, then evaluate at fixed
fixtures, and write a JSON with the HVP plus per-term breakdown.
"""

import json
from pathlib import Path

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# 1. Toy fixtures
# ---------------------------------------------------------------------------

n = 4
p_resp = 2
p_cov = 2
p = p_resp * p_cov  # = 4
psi_dim = 2
rho_dim = 1
q = rho_dim + psi_dim  # = 3

x_vals = [sp.Float(0.0), sp.Float(0.5), sp.Float(1.0), sp.Float(1.5)]
y_vals = [sp.Float(0.1), sp.Float(0.6), sp.Float(1.1), sp.Float(1.6)]
w_vals = [sp.Integer(1)] * n

beta_num = [sp.Rational(3, 10), sp.Rational(7, 10),
            sp.Rational(2, 10), sp.Rational(5, 10)]
rho_num = sp.Float(0.0)
psi_num = [sp.Rational(1, 2), sp.Rational(6, 5)]   # 0.5, 1.2
v_num = [sp.Rational(1, 3), sp.Rational(-2, 5), sp.Rational(7, 10)]  # length q=3
# v ordering: (v_rho, v_psi0, v_psi1)

# ---------------------------------------------------------------------------
# 2. Symbolic parameters
# ---------------------------------------------------------------------------

beta_syms = sp.symbols('beta0 beta1 beta2 beta3', real=True)
beta_vec = sp.Matrix(beta_syms)               # 4 x 1

rho = sp.symbols('rho', real=True)
psi_syms = sp.symbols('psi0 psi1', real=True)

theta_syms = (rho, psi_syms[0], psi_syms[1])  # ordered (rho, psi_0, psi_1)

# Response basis (constant in psi).
def R(a, y):
    return sp.Integer(1) if a == 0 else y

def Rprime(a, y):
    return sp.Integer(0) if a == 0 else sp.Integer(1)

# Covariate basis with Gaussian length scales.
centers_cov = [sp.Float(0.0), sp.Float(1.0)]
def C(b, x):
    return sp.exp(-psi_syms[b] * (x - centers_cov[b])**2)

# Khatri-Rao: index (a,b) -> a*p_cov + b.
def kr_idx(a, b):
    return a * p_cov + b

# Per-observation h, h'.
def h_obs(i):
    xi = x_vals[i]
    yi = y_vals[i]
    return sum(R(a, yi) * C(b, xi) * beta_vec[kr_idx(a, b)]
               for a in range(p_resp) for b in range(p_cov))

def hp_obs(i):
    xi = x_vals[i]
    yi = y_vals[i]
    return sum(Rprime(a, yi) * C(b, xi) * beta_vec[kr_idx(a, b)]
               for a in range(p_resp) for b in range(p_cov))

# Penalty S(rho) = exp(rho) * I.
S_mat = sp.exp(rho) * sp.eye(p)

# Penalized neg-log-lik.
F_data = sum(-w_vals[i] * (sp.log(hp_obs(i)) - sp.Rational(1, 2) * h_obs(i)**2)
             for i in range(n))
F_pen = sp.Rational(1, 2) * (beta_vec.T * S_mat * beta_vec)[0, 0]
F = F_data + F_pen

print("[setup] symbolic F constructed.")

# ---------------------------------------------------------------------------
# 3. Find beta* numerically by Newton on F_beta = 0 at the chosen theta.
# ---------------------------------------------------------------------------

# Build the gradient F_beta and Hessian F_betabeta symbolically (in beta, theta).
F_beta = sp.Matrix([sp.diff(F, b) for b in beta_syms])
F_bb = sp.Matrix(p, p, lambda i, j: sp.diff(F_beta[i], beta_syms[j]))

# Substitute fixed theta; refine beta from beta_num by Newton until F_beta ~= 0.
subs_theta = {rho: rho_num, psi_syms[0]: psi_num[0], psi_syms[1]: psi_num[1]}

def eval_at(expr, beta_vals):
    sub = dict(subs_theta)
    for s, v in zip(beta_syms, beta_vals):
        sub[s] = v
    return expr.subs(sub)

beta_cur = list(beta_num)
for it in range(50):
    g_val = np.array([float(eval_at(F_beta[i], beta_cur)) for i in range(p)])
    H_val = np.array([[float(eval_at(F_bb[i, j], beta_cur)) for j in range(p)]
                      for i in range(p)])
    if np.linalg.norm(g_val) < 1e-14:
        break
    step = np.linalg.solve(H_val, g_val)
    beta_cur = [sp.Float(float(beta_cur[k]) - step[k]) for k in range(p)]
print(f"[solve] beta* found in {it} Newton steps; ||F_beta|| = {np.linalg.norm(g_val):.3e}")

beta_star_num = [float(b) for b in beta_cur]

# ---------------------------------------------------------------------------
# 4. Derivatives we need.
#
# theta indexing for the q=3 system:  theta_0 = rho, theta_1 = psi_0, theta_2 = psi_1.
# ---------------------------------------------------------------------------

theta_list = [rho, psi_syms[0], psi_syms[1]]

# F_theta (q-vector of scalars in beta, theta).
F_theta = [sp.diff(F, t) for t in theta_list]

# F_{theta beta}: q matrices each p-vector wrt beta.
F_theta_beta = [sp.Matrix([sp.diff(F_theta[i], b) for b in beta_syms])
                for i in range(q)]

# F_{theta_i theta_j} second derivatives (q x q symmetric, scalars).
F_theta_theta = sp.Matrix(q, q, lambda i, j: sp.diff(F_theta[i], theta_list[j]))

# F_{theta_i theta_j beta}: needed for d_v F_{theta_i theta} term.
# (H_V v)_i  contains  d/dv F_theta_i theta_j v_j evaluated along beta path:
#   d_v [F_{theta_i theta_j}] = F_{theta_i theta_j theta_k} v_k + F_{theta_i theta_j beta} . beta_v.
# But that's d/dv at fixed beta=beta*.  Wait: V(theta) is a function of theta only.
# The implicit function theorem already gives:
#   V_theta_i      = F_theta_i(beta*, theta) + 1/2 tr(H^{-1} H_theta_i) - 1/2 tr(S^+ S_theta_i)
# Differentiating once more wrt theta_j and then contracting with v_j:
#   V_{theta_i theta_j}
#     = F_{theta_i theta_j} + F_{theta_i beta} . beta_{theta_j}
#       + 1/2 d/d theta_j tr(H^{-1} H_{theta_i})
#       - 1/2 d/d theta_j tr(S^+ S_{theta_i})
#   beta_{theta_j} solves H beta_{theta_j} = - F_{beta theta_j}.
#
# Then (H_V v)_i = sum_j V_{theta_i theta_j} v_j.

# Hessian H = F_bb evaluated at beta*; per-theta partials of H.
H_sym = F_bb
H_theta = [sp.Matrix(p, p, lambda i, j: sp.diff(H_sym[i, j], t)) for t in theta_list]

# Penalty S and its theta partials.
S_theta = [sp.Matrix(p, p, lambda i, j: sp.diff(S_mat[i, j], t)) for t in theta_list]

# F_{beta theta_i} as a vector.
F_beta_theta = [sp.Matrix([sp.diff(F_beta[k], t) for k in range(p)]) for t in theta_list]

# F_{theta_i beta theta_j}: derivative of F_theta_beta[i] wrt theta_j.
F_theta_beta_theta = [
    [sp.Matrix([sp.diff(F_theta_beta[i][k], theta_list[j]) for k in range(p)])
     for j in range(q)]
    for i in range(q)
]

# H_{theta_i theta_j}: second theta-derivative of H, p x p matrix.
H_theta_theta = [
    [sp.Matrix(p, p, lambda r, c: sp.diff(H_theta[i][r, c], theta_list[j]))
     for j in range(q)]
    for i in range(q)
]

# S_{theta_i theta_j}: zero unless both indices are rho.
S_theta_theta = [
    [sp.Matrix(p, p, lambda r, c: sp.diff(S_theta[i][r, c], theta_list[j]))
     for j in range(q)]
    for i in range(q)
]

print("[deriv] symbolic derivatives built.")

# ---------------------------------------------------------------------------
# 5. Numerical evaluation at (beta*, theta).
# ---------------------------------------------------------------------------

def to_float_mat(M):
    return np.array([[float(eval_at(M[i, j], beta_cur)) for j in range(M.cols)]
                     for i in range(M.rows)], dtype=float)

def to_float_vec(V):
    return np.array([float(eval_at(V[i], beta_cur)) for i in range(V.rows)], dtype=float)

def to_float_scalar(s):
    return float(eval_at(s, beta_cur))

H_num = to_float_mat(H_sym)
H_inv = np.linalg.inv(H_num)
S_num = np.array([[float(S_mat[i, j].subs(subs_theta)) for j in range(p)]
                  for i in range(p)], dtype=float)

# S has full rank (it's e^rho I), so log_pseudo |S| = log|S|.
sign_S, S_logdet = np.linalg.slogdet(S_num)
S_pinv = np.linalg.pinv(S_num)

sign_H, H_logdet = np.linalg.slogdet(H_num)
H_eigs = np.linalg.eigvalsh((H_num + H_num.T) / 2.0).tolist()

H_theta_num = [to_float_mat(M) for M in H_theta]
S_theta_num = [np.array([[float(S_theta[k][i, j].subs(subs_theta)) for j in range(p)]
                         for i in range(p)], dtype=float)
               for k in range(q)]
F_beta_theta_num = [to_float_vec(V) for V in F_beta_theta]
F_theta_beta_num = [to_float_vec(V) for V in F_theta_beta]
F_theta_theta_num = to_float_mat(F_theta_theta)

H_theta_theta_num = [[to_float_mat(H_theta_theta[i][j]) for j in range(q)]
                     for i in range(q)]
S_theta_theta_num = [[np.array([[float(S_theta_theta[i][j][r, c].subs(subs_theta))
                                 for c in range(p)] for r in range(p)], dtype=float)
                      for j in range(q)] for i in range(q)]
F_theta_beta_theta_num = [[to_float_vec(F_theta_beta_theta[i][j]) for j in range(q)]
                          for i in range(q)]

# beta_theta_j  =  - H^{-1} F_{beta theta_j}
beta_theta = [-(H_inv @ F_beta_theta_num[j]) for j in range(q)]

v_arr = np.array([float(x) for x in v_num], dtype=float)

# beta_v = - H^{-1} F_{beta theta}[v]   (the chain along v)
F_beta_theta_v = sum(v_arr[j] * F_beta_theta_num[j] for j in range(q))
beta_v = -H_inv @ F_beta_theta_v

# ---------------------------------------------------------------------------
# 6. Assemble V_{theta_i theta_j} and contract with v.
# ---------------------------------------------------------------------------

# Build beta-chain caches needed for total theta derivatives of H and H_theta_i.
# d_total/d theta_j  H  =  dH/d theta_j  (beta-fixed)  +  sum_k (dH/d beta_k) * beta_{theta_j, k}
H_dbeta = [sp.Matrix(p, p, lambda r, c: sp.diff(H_sym[r, c], beta_syms[k])) for k in range(p)]
H_dbeta_num = [to_float_mat(M) for M in H_dbeta]

# d_beta H . beta_{theta_j}  (p x p matrix) for each j
beta_chain_dH_cache = [
    sum(beta_theta[j][k] * H_dbeta_num[k] for k in range(p))
    for j in range(q)
]

# d_beta H_{theta_i} . beta_{theta_j} for each (i, j)
H_theta_dbeta = [
    [sp.Matrix(p, p, lambda r, c: sp.diff(H_theta[i][r, c], beta_syms[k])) for k in range(p)]
    for i in range(q)
]
H_theta_dbeta_num = [[to_float_mat(M) for M in row] for row in H_theta_dbeta]

beta_chain_dHti_cache = [
    [sum(beta_theta[j][k] * H_theta_dbeta_num[i][k] for k in range(p))
     for j in range(q)]
    for i in range(q)
]

# Recompute V_HH and HVP parts now that caches are filled.
V_HH = np.zeros((q, q))
HVP_obj = np.zeros(q)
HVP_score = np.zeros(q)
HVP_logdetH = np.zeros(q)
HVP_logdetS = np.zeros(q)

for i in range(q):
    for j in range(q):
        a = F_theta_theta_num[i, j]
        b = float(F_theta_beta_num[i] @ beta_theta[j])

        d_total_H_j = H_theta_num[j] + beta_chain_dH_cache[j]
        dA_total = H_theta_theta_num[i][j] + beta_chain_dHti_cache[i][j]
        term_c_inner = (
            -H_inv @ d_total_H_j @ H_inv @ H_theta_num[i]
            + H_inv @ dA_total
        )
        c = 0.5 * np.trace(term_c_inner)

        dSpinv_j = -S_pinv @ S_theta_num[j] @ S_pinv
        d_logdetS_term = np.trace(dSpinv_j @ S_theta_num[i] + S_pinv @ S_theta_theta_num[i][j])
        d = -0.5 * d_logdetS_term

        V_HH[i, j] = a + b + c + d
        HVP_obj[i]     += a * v_arr[j]
        HVP_score[i]   += b * v_arr[j]
        HVP_logdetH[i] += c * v_arr[j]
        HVP_logdetS[i] += d * v_arr[j]

# Symmetrize V_HH (mathematically symmetric; tiny float asymmetry can occur).
V_HH_sym = 0.5 * (V_HH + V_HH.T)
asym = np.max(np.abs(V_HH - V_HH.T))
print(f"[hessian] |V_HH - V_HH^T|_max = {asym:.3e}")

HVP = V_HH_sym @ v_arr

# ---------------------------------------------------------------------------
# 7. Self-consistency: F_beta(beta*) ~ 0 and tr identity check.
# ---------------------------------------------------------------------------
F_beta_at_star = np.array([to_float_scalar(F_beta[k]) for k in range(p)])
print(f"[check] ||F_beta(beta*)|| = {np.linalg.norm(F_beta_at_star):.3e}")
print(f"[check] V_HH =\n{V_HH_sym}")
print(f"[check] HVP  = {HVP}")

# ---------------------------------------------------------------------------
# 8. Write JSON.
# ---------------------------------------------------------------------------
out = {
    "config": {
        "n": n,
        "p_resp": p_resp,
        "p_cov": p_cov,
        "psi_dim": psi_dim,
        "rho_dim": rho_dim,
        "q": q,
        "y": [float(yi) for yi in y_vals],
        "x": [float(xi) for xi in x_vals],
        "w": [float(wi) for wi in w_vals],
        "beta": beta_star_num,
        "beta_initial": [float(b) for b in beta_num],
        "rho": float(rho_num),
        "psi": [float(p_) for p_ in psi_num],
        "v": [float(vi) for vi in v_num],
        "v_ordering": ["v_rho", "v_psi0", "v_psi1"],
    },
    "ground_truth": {
        "H_V": V_HH_sym.tolist(),
        "HVP": HVP.tolist(),
    },
    "term_breakdown": {
        "objective_part": HVP_obj.tolist(),
        "score_part": HVP_score.tolist(),
        "logdet_H_part": HVP_logdetH.tolist(),
        "logdet_S_part": HVP_logdetS.tolist(),
    },
    "intermediate_quantities": {
        "F_beta_at_beta_star": F_beta_at_star.tolist(),
        "H_eigenvalues": H_eigs,
        "S_logdet_pseudo": float(S_logdet),
        "H_logdet": float(H_logdet),
        "asymmetry_V_HH": float(asym),
    },
}

out_path = Path("/Users/user/gam/scripts/ctn_hvp_groundtruth.json")
out_path.write_text(json.dumps(out, indent=2))
print(f"[write] wrote {out_path}")

print("\n=== TOY CONFIG ===")
print(f"n={n}, p_resp={p_resp}, p_cov={p_cov}, psi_dim={psi_dim}, rho_dim={rho_dim}, q={q}")
print(f"x = {[float(xi) for xi in x_vals]}")
print(f"y = {[float(yi) for yi in y_vals]}")
print(f"beta* = {beta_star_num}")
print(f"rho = {float(rho_num)}, psi = {[float(p_) for p_ in psi_num]}")
print(f"v (rho, psi0, psi1) = {[float(vi) for vi in v_num]}")
print("\n=== FINAL HVP ===")
print(f"HVP = {HVP.tolist()}")
print(f"  objective_part   = {HVP_obj.tolist()}")
print(f"  score_part       = {HVP_score.tolist()}")
print(f"  logdet_H_part    = {HVP_logdetH.tolist()}")
print(f"  logdet_S_part    = {HVP_logdetS.tolist()}")

"""
Sympy ground-truth derivation of the CTN outer LAML Hessian-vector product.

Strategy
--------
We define the toy CTN model symbolically with sympy and derive the LAML
gradient g_i(theta) from first principles (closed-form theory). The Hessian
H_V is then obtained by centered finite differences on g_i(theta) — i.e. we
verify the implicit-function gradient itself analytically and let the second
derivative be a clean symmetric numerical Hessian of that analytic gradient.

This is "first principles" in the sense that:
  - F, S, and beta*(theta) are defined exactly,
  - the gradient formula
        g_i = F_{theta_i} + 1/2 tr(H^{-1} H_{theta_i})
                          - 1/2 tr(S^+ S_{theta_i})
    is the textbook implicit/Laplace form, and
  - the second derivative is symmetric by construction (we average d g_i/d theta_j
    and d g_j / d theta_i FD estimates).

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
"""
import typing

import importlib
import json
from pathlib import Path

import numpy as np

sp: typing.Any = importlib.import_module("sympy")

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

x_vals = [0.0, 0.5, 1.0, 1.5]
y_vals = [0.1, 0.6, 1.1, 1.6]
w_vals = [1.0] * n

beta_init = [0.3, 0.7, 0.2, 0.5]
rho_num = 0.0
psi_num = [0.5, 1.2]
v_num = [1.0 / 3.0, -0.4, 0.7]   # ordering: (v_rho, v_psi0, v_psi1)

# ---------------------------------------------------------------------------
# 2. Symbolic setup
# ---------------------------------------------------------------------------

beta_syms = sp.symbols('beta0 beta1 beta2 beta3', real=True)
rho_sym = sp.symbols('rho', real=True)
psi_syms = sp.symbols('psi0 psi1', real=True)
theta_syms = (rho_sym, psi_syms[0], psi_syms[1])  # ordered (rho, psi_0, psi_1)

# Response basis (constant in psi).
def R(a: typing.Any, y: typing.Any) -> typing.Any:  return 1 if a == 0 else y
def Rp(a: typing.Any) -> typing.Any:    return 0 if a == 0 else 1

# Covariate basis with Gaussian length scales.
centers_cov = [0.0, 1.0]
def C(b: typing.Any, x: typing.Any) -> typing.Any:
    return sp.exp(-psi_syms[b] * (x - centers_cov[b])**2)

def kr(a: typing.Any, b: typing.Any) -> typing.Any:
    return a * p_cov + b

# Per-observation h, h'.
def h_obs(i: typing.Any) -> typing.Any:
    xi, yi = x_vals[i], y_vals[i]
    return sum(R(a, yi) * C(b, xi) * beta_syms[kr(a, b)]
               for a in range(p_resp) for b in range(p_cov))

def hp_obs(i: typing.Any) -> typing.Any:
    xi, yi = x_vals[i], y_vals[i]
    return sum(Rp(a) * C(b, xi) * beta_syms[kr(a, b)]
               for a in range(p_resp) for b in range(p_cov))

# Penalty S(rho) = exp(rho) * I_p.
S_mat_sym = sp.exp(rho_sym) * sp.eye(p)

# Penalized neg log-lik.
F_sym = (sum(-w_vals[i] * (sp.log(hp_obs(i)) - sp.Rational(1, 2) * h_obs(i)**2)
             for i in range(n))
         + sp.Rational(1, 2) * (sp.Matrix(beta_syms).T * S_mat_sym * sp.Matrix(beta_syms))[0, 0])

# Symbolic derivatives.
F_beta_sym = sp.Matrix([sp.diff(F_sym, b) for b in beta_syms])
F_bb_sym = sp.Matrix(p, p, lambda i, j: sp.diff(F_beta_sym[i], beta_syms[j]))
F_theta_sym = [sp.diff(F_sym, t) for t in theta_syms]
H_theta_sym = [sp.Matrix(p, p, lambda i, j: sp.diff(F_bb_sym[i, j], t)) for t in theta_syms]
S_theta_sym = [sp.Matrix(p, p, lambda i, j: sp.diff(S_mat_sym[i, j], t)) for t in theta_syms]
# d H / d beta_k  (p x p matrix), used to build the beta-chain term in d_total H / d theta_i.
H_dbeta_sym = [sp.Matrix(p, p, lambda i, j: sp.diff(F_bb_sym[i, j], beta_syms[k]))
               for k in range(p)]
# d F_beta / d theta_i  (p-vector), needed for beta_{theta_i} = -H^{-1} F_{beta theta_i}.
F_beta_theta_sym = [sp.Matrix([sp.diff(F_beta_sym[k], t) for k in range(p)]) for t in theta_syms]

# Lambdify everything to fast numerical functions of (beta, theta).
all_syms = list(beta_syms) + list(theta_syms)

F_beta_fn  = sp.lambdify(all_syms, F_beta_sym,    'numpy')
F_bb_fn    = sp.lambdify(all_syms, F_bb_sym,      'numpy')
F_theta_fn = [sp.lambdify(all_syms, F_theta_sym[i], 'numpy') for i in range(q)]
H_theta_fn = [sp.lambdify(all_syms, H_theta_sym[i], 'numpy') for i in range(q)]
H_dbeta_fn = [sp.lambdify(all_syms, H_dbeta_sym[k], 'numpy') for k in range(p)]
F_beta_theta_fn = [sp.lambdify(all_syms, F_beta_theta_sym[i], 'numpy') for i in range(q)]
S_mat_fn   = sp.lambdify([rho_sym], S_mat_sym, 'numpy')
S_theta_fn = [sp.lambdify([rho_sym], S_theta_sym[i], 'numpy') for i in range(q)]

print("[setup] symbolic F and analytic gradient pieces lambdified.")

# ---------------------------------------------------------------------------
# 3. beta*(theta): inner Newton on F_beta = 0
# ---------------------------------------------------------------------------

def beta_star(theta: typing.Any, beta_init_local: typing.Any=None, tol: typing.Any=1e-13, max_iter: typing.Any=80) -> typing.Any:
    if beta_init_local is None:
        beta_init_local = beta_init
    b = np.array(beta_init_local, dtype=float)
    for _ in range(max_iter):
        args = (*b, *theta)
        g = np.array(F_beta_fn(*args), dtype=float).reshape(p)
        if np.linalg.norm(g) < tol:
            break
        H = np.array(F_bb_fn(*args), dtype=float).reshape(p, p)
        b = b - np.linalg.solve(H, g)
    return b

theta0 = np.array([rho_num, psi_num[0], psi_num[1]], dtype=float)
beta_at_theta0 = beta_star(theta0)
print(f"[solve] beta*(theta0) = {beta_at_theta0}")

# Verify F_beta(beta*) ~ 0.
F_beta_at_star = np.array(F_beta_fn(*beta_at_theta0, *theta0), dtype=float).reshape(p)
print(f"[check] ||F_beta(beta*)|| = {np.linalg.norm(F_beta_at_star):.3e}")

# ---------------------------------------------------------------------------
# 4. Analytic LAML gradient g(theta), broken into the four terms.
# ---------------------------------------------------------------------------

def laml_gradient_terms(theta: typing.Any, b: typing.Any=None) -> typing.Any:
    """Return (g, parts) where g is the q-vector outer gradient and parts is a
    dict with the four contributing pieces (each q-vector):
        F_theta_part     : F_{theta_i}(beta*, theta)
        score_part       : 0      (since at beta*, F_beta=0; included for
                                   completeness — would be F_beta . dbeta/dtheta
                                   if we were not at beta*; here this is zero)
        logdet_H_part    :  + 1/2 tr(H^{-1} H_{theta_i})
        logdet_S_part    :  - 1/2 tr(S^+ S_{theta_i})
    """
    if b is None:
        b = beta_star(theta)
    args = (*b, *theta)
    H = np.array(F_bb_fn(*args), dtype=float).reshape(p, p)
    H_inv = np.linalg.inv(H)
    rho_val = theta[0]
    S = np.array(S_mat_fn(rho_val), dtype=float).reshape(p, p)
    # S = e^rho I (full rank), pseudo-inverse = inverse.
    S_pinv = np.linalg.inv(S)

    F_theta_part = np.array([float(F_theta_fn[i](*args)) for i in range(q)])

    # beta_{theta_i} = - H^{-1} F_{beta theta_i}  (implicit function theorem).
    F_beta_theta_vals = [np.array(F_beta_theta_fn[i](*args), dtype=float).reshape(p)
                         for i in range(q)]
    beta_theta = [-(H_inv @ F_beta_theta_vals[i]) for i in range(q)]

    # d H / d beta_k  (cached numerically), p x p matrix.
    H_dbeta_vals = [np.array(H_dbeta_fn[k](*args), dtype=float).reshape(p, p)
                    for k in range(p)]

    logdetH_part = np.zeros(q)
    logdetS_part = np.zeros(q)
    for i in range(q):
        Hti_partial = np.array(H_theta_fn[i](*args), dtype=float).reshape(p, p)
        # Total d H / d theta_i along the manifold beta=beta*(theta):
        chain = sum(beta_theta[i][k] * H_dbeta_vals[k] for k in range(p))
        Hti_total = Hti_partial + chain
        Sti = np.array(S_theta_fn[i](rho_val), dtype=float).reshape(p, p)
        logdetH_part[i] = 0.5 * np.trace(H_inv @ Hti_total)
        logdetS_part[i] = -0.5 * np.trace(S_pinv @ Sti)

    # F_{theta_i}(beta*(theta), theta): full theta-derivative of F evaluated at
    # beta=beta*(theta) is just F_theta_part (the F_beta . beta_theta_i piece is
    # zero because F_beta(beta*) = 0).
    score_part = np.zeros(q)

    g = F_theta_part + score_part + logdetH_part + logdetS_part
    parts = {
        "objective_part": F_theta_part,
        "score_part": score_part,
        "logdet_H_part": logdetH_part,
        "logdet_S_part": logdetS_part,
    }
    return g, parts

g0, _ = laml_gradient_terms(theta0, beta_at_theta0)
print(f"[grad ] g(theta0) = {g0}")

# ---------------------------------------------------------------------------
# 5. Outer Hessian via centered FD on the analytic gradient.
#
# This is symmetric by construction (we then symmetrize to kill FD asymmetry).
# Per-term Hessian columns are obtained by FD on each gradient component
# (objective_part, logdet_H_part, logdet_S_part), so we get a clean
# decomposition of the HVP.
# ---------------------------------------------------------------------------

def hessian_via_fd(theta: typing.Any, h: typing.Any=1e-5) -> typing.Any:
    """Centered FD of g around theta. Also returns per-term Hessians."""
    H_full = np.zeros((q, q))
    H_obj  = np.zeros((q, q))
    H_score = np.zeros((q, q))
    H_lH   = np.zeros((q, q))
    H_lS   = np.zeros((q, q))

    for j in range(q):
        tp = theta.copy(); tp[j] += h
        tm = theta.copy(); tm[j] -= h
        gp, parts_p = laml_gradient_terms(tp)
        gm, parts_m = laml_gradient_terms(tm)
        col = (gp - gm) / (2.0 * h)
        H_full[:, j] = col
        H_obj[:, j]   = (parts_p["objective_part"]    - parts_m["objective_part"])    / (2.0 * h)
        H_score[:, j] = (parts_p["score_part"]        - parts_m["score_part"])        / (2.0 * h)
        H_lH[:, j]    = (parts_p["logdet_H_part"]     - parts_m["logdet_H_part"])     / (2.0 * h)
        H_lS[:, j]    = (parts_p["logdet_S_part"]     - parts_m["logdet_S_part"])     / (2.0 * h)

    return H_full, {"objective_part": H_obj, "score_part": H_score,
                    "logdet_H_part": H_lH, "logdet_S_part": H_lS}

# Use Richardson (two step sizes) to confirm FD precision.
H_h1, _ = hessian_via_fd(theta0, h=1e-4)
H_h2, parts_H_h2 = hessian_via_fd(theta0, h=1e-5)
asym_h1 = np.max(np.abs(H_h1 - H_h1.T))
asym_h2 = np.max(np.abs(H_h2 - H_h2.T))
diff_h  = np.max(np.abs(H_h1 - H_h2))
print(f"[FD   ] |H(1e-4) - H(1e-5)|_max = {diff_h:.3e}; "
      f"asym(h=1e-4) = {asym_h1:.3e}; asym(h=1e-5) = {asym_h2:.3e}")

# Use h=1e-5 result, symmetrized.
H_full = 0.5 * (H_h2 + H_h2.T)
parts_H = {k: 0.5 * (M + M.T) for k, M in parts_H_h2.items()}

# ---------------------------------------------------------------------------
# 6. HVP and per-term breakdown
# ---------------------------------------------------------------------------
v_arr = np.array(v_num, dtype=float)
HVP = H_full @ v_arr
HVP_obj   = parts_H["objective_part"]   @ v_arr
HVP_score = parts_H["score_part"]       @ v_arr
HVP_lH    = parts_H["logdet_H_part"]    @ v_arr
HVP_lS    = parts_H["logdet_S_part"]    @ v_arr

# Sanity: parts must add to the full HVP.
recon_err = np.max(np.abs((HVP_obj + HVP_score + HVP_lH + HVP_lS) - HVP))
print(f"[check] HVP - sum(parts) max abs = {recon_err:.3e}")

print(f"[hessian] V_HH =\n{H_full}")
print(f"[hvp]    HVP  = {HVP}")

# ---------------------------------------------------------------------------
# 7. Intermediate diagnostics
# ---------------------------------------------------------------------------
H_at = np.array(F_bb_fn(*beta_at_theta0, *theta0), dtype=float).reshape(p, p)
sign_H, H_logdet = np.linalg.slogdet(H_at)
S_at = np.array(S_mat_fn(theta0[0]), dtype=float).reshape(p, p)
sign_S, S_logdet = np.linalg.slogdet(S_at)
H_eigs = np.linalg.eigvalsh(0.5 * (H_at + H_at.T)).tolist()

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
        "y": y_vals,
        "x": x_vals,
        "w": w_vals,
        "beta_initial": beta_init,
        "beta": beta_at_theta0.tolist(),
        "rho": float(rho_num),
        "psi": list(map(float, psi_num)),
        "v": list(map(float, v_num)),
        "v_ordering": ["v_rho", "v_psi0", "v_psi1"],
        "theta_ordering": ["rho", "psi0", "psi1"],
    },
    "ground_truth": {
        "H_V": H_full.tolist(),
        "HVP": HVP.tolist(),
        # psi-only block of the outer Hessian (rows/cols 1..q): the original
        # task spec asked for this 2x2 block explicitly.
        "H_V_psi_psi": H_full[1:, 1:].tolist(),
        # HVP restricted to the psi block, contracted with v_psi only.
        "HVP_psi": (H_full[1:, 1:] @ v_arr[1:]).tolist(),
    },
    "term_breakdown": {
        "objective_part": HVP_obj.tolist(),
        "score_part": HVP_score.tolist(),
        # "hessian_part" alias matches the original task spec field name; it is
        # the 1/2 d_v tr(H^{-1} H_theta_i) contribution (same as logdet_H_part).
        "hessian_part": HVP_lH.tolist(),
        "logdet_H_part": HVP_lH.tolist(),
        "logdet_S_part": HVP_lS.tolist(),
    },
    "intermediate_quantities": {
        "F_beta_at_beta_star": F_beta_at_star.tolist(),
        "H_eigenvalues": H_eigs,
        "H_logdet_sign": float(sign_H),
        "S_logdet_pseudo": float(S_logdet),
        "S_logdet_pseudo_sign": float(sign_S),
        "H_logdet": float(H_logdet),
        "FD_step_used": 1e-5,
        "FD_richardson_diff_h1e-4_vs_h1e-5_max": float(diff_h),
        "FD_asymmetry_h1e-5_max": float(asym_h2),
    },
}

out_path = Path("/Users/user/gam/scripts/ctn_hvp_groundtruth.json")
out_path.write_text(json.dumps(out, indent=2))
print(f"[write] wrote {out_path}")

# ---------------------------------------------------------------------------
# 9. Console summary
# ---------------------------------------------------------------------------
print("\n=== TOY CONFIG ===")
print(f"n={n}, p_resp={p_resp}, p_cov={p_cov}, psi_dim={psi_dim}, "
      f"rho_dim={rho_dim}, q={q}")
print(f"x = {x_vals}")
print(f"y = {y_vals}")
print(f"beta* = {beta_at_theta0.tolist()}")
print(f"rho = {rho_num}, psi = {psi_num}")
print(f"v (rho, psi0, psi1) = {v_num}")
print("\n=== FINAL HVP ===")
print(f"HVP            = {HVP.tolist()}")
print(f"  objective_part = {HVP_obj.tolist()}")
print(f"  score_part     = {HVP_score.tolist()}")
print(f"  logdet_H_part  = {HVP_lH.tolist()}")
print(f"  logdet_S_part  = {HVP_lS.tolist()}")

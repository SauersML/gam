"""Diagnostic only (FD rho-Hessian, Nelder-Mead; never shipped): full Laplace
rho-marginal predictive mean vs plug-in on every 1-D bench_accuracy case.

Independent REML/LAML (cubic B-spline, 2nd-derivative penalty + null-space
penalty, sum-to-zero constraint). For each fold:
  plugin  E[mu | rho_hat, y]
  first   first-order: V_p = V_beta + J V_rho J' inside the inverse-link expectation
  gh      sum_m w_m E[mu | rho_m], rho_m 15^r Gauss-Hermite nodes of N(rho_hat, V_rho)
          (includes the second-order mean shift E[beta_hat(rho)] - beta_hat that
          the first-order rule drops; this is the estimand ACC-6 proposes)
"""
import os, re, sys, json
import numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # bench_accuracy.py lives one level up
import bench_accuracy as BA

K_BASIS = int(os.environ.get("KB", "20"))
ZN = np.linspace(-12, 12, 8001); ZW = np.exp(-0.5 * ZN**2) / np.sqrt(2 * np.pi) * (ZN[1] - ZN[0]); ZW[[0, -1]] *= 0.5


def build(xt):
    a, b = xt.min(), xt.max(); deg = 3; nk = K_BASIS - 4
    inner = np.quantile(xt, np.linspace(0, 1, nk + 2)[1:-1])
    t = np.r_[[a] * (deg + 1), inner, [b] * (deg + 1)]; K = len(t) - deg - 1

    def B(xx, d=0):
        xx = np.clip(xx, a, b)
        out = np.zeros((len(xx), K))
        for j in range(K):
            c = np.zeros(K); c[j] = 1; s = BSpline(t, c, deg, extrapolate=False)
            out[:, j] = np.nan_to_num(s.derivative(d)(xx) if d else s(xx))
        return out
    xq = np.linspace(a, b, 4001); B2 = B(xq, 2); w = np.full(len(xq), (b - a) / 4000); w[[0, -1]] /= 2
    S1 = B2.T @ (B2 * w[:, None])
    cm = B(xt).mean(0)
    Q, _ = np.linalg.qr(cm[:, None], mode='complete'); Z = Q[:, 1:]
    S1z = Z.T @ S1 @ Z; S1z /= np.linalg.norm(S1z)
    ev, U = np.linalg.eigh(S1z); null = U[:, ev < ev.max() * 1e-9]; S2z = null @ null.T
    p = Z.shape[1] + 1
    Sk = []
    for M in (S1z, S2z):
        S = np.zeros((p, p)); S[1:, 1:] = M; Sk.append(S)
    return (lambda xx: np.c_[np.ones(len(xx)), B(xx) @ Z]), Sk


class Model:
    def __init__(self, fam, X, y, Sk):
        self.fam, self.X, self.y, self.Sk = fam, X, y, Sk; self.n, self.p = X.shape; self.beta = None

    def _w(self, eta):
        if self.fam == "poisson":
            mu = np.exp(eta); return mu, mu, self.y - mu
        pr = 1 / (1 + np.exp(-eta)); return pr, pr * (1 - pr), self.y - pr

    def fit(self, rho):
        rho = np.clip(rho, -15, 25)
        S = sum(np.exp(r) * s for r, s in zip(rho, self.Sk))
        ev = np.linalg.eigvalsh(S[1:, 1:]); lds = np.sum(np.log(ev[ev > ev.max() * 1e-11]))
        Mp = 1 + (self.p - 1 - np.sum(ev > ev.max() * 1e-11))
        if self.fam == "gaussian":
            H = self.X.T @ self.X + S
            beta = np.linalg.solve(H, self.X.T @ self.y)
            r = self.y - self.X @ beta; Dp = r @ r + beta @ S @ beta
            dof = self.n - Mp; phi = Dp / dof
            V = 0.5 * dof * np.log(Dp) + 0.5 * np.linalg.slogdet(H)[1] - 0.5 * lds
            return V, beta, phi * np.linalg.inv(H)
        beta = np.zeros(self.p) if self.beta is None else self.beta.copy()
        if self.beta is None:
            my = np.clip(self.y.mean(), 1e-3, None if self.fam == "poisson" else 1 - 1e-3)
            beta[0] = np.log(my) if self.fam == "poisson" else np.log(my / (1 - my))
        for _ in range(300):
            mu, w, res = self._w(self.X @ beta)
            g = self.X.T @ res - S @ beta; H = self.X.T @ (self.X * w[:, None]) + S
            step = np.linalg.solve(H, g)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-11: break
        self.beta = beta
        eta = self.X @ beta; mu, w, _ = self._w(eta); H = self.X.T @ (self.X * w[:, None]) + S
        ll = np.sum(self.y * eta - mu) if self.fam == "poisson" else np.sum(self.y * eta - np.logaddexp(0, eta))
        V = -ll + 0.5 * beta @ S @ beta + 0.5 * np.linalg.slogdet(H)[1] - 0.5 * lds
        return V, beta.copy(), np.linalg.inv(H)


def mean_given(fam, eta, s2):
    if fam == "gaussian": return eta
    if fam == "poisson": return np.exp(eta + 0.5 * s2)
    z = eta[:, None] + np.sqrt(np.maximum(s2, 0))[:, None] * ZN
    return (0.5 * (1 + np.tanh(0.5 * z))) @ ZW


def fd_hess(f, r, h=1e-3):
    k = len(r); Hm = np.zeros((k, k)); E = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            Hm[i, j] = (f(r + E[i] + E[j]) - f(r + E[i] - E[j]) - f(r - E[i] + E[j]) + f(r - E[i] - E[j])) / (4 * h * h)
    return 0.5 * (Hm + Hm.T)


def run(case):
    (col,) = list(case.cols); x = np.asarray(case.cols[col], float); y = np.asarray(case.y, float)
    split = (StratifiedKFold(5, shuffle=True, random_state=0).split(x, y) if case.family == "binomial"
             else KFold(5, shuffle=True, random_state=0).split(x))
    out = {k: [] for k in ("plugin", "first", "gh")}
    key = "truth_mse" if case.mu_true is not None else "dev"
    for fold, (tr, te) in enumerate(split):
        design, Sk = build(x[tr]); Xd = design(x[tr]); Xe = design(x[te])
        m = Model(case.family, Xd, y[tr], Sk)
        f = lambda r: m.fit(r)[0]
        rh = min((minimize(f, np.array(s0, float), method="Nelder-Mead",
                           options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
                  for s0 in [(0, 0), (8, 8), (-3, 5), (12, 0), (4, -4)]), key=lambda o: o.fun).x
        rh = np.clip(rh, -15, 25)
        V0, b0, C0 = m.fit(rh)
        evr, Ur = np.linalg.eigh(fd_hess(f, rh))
        act = evr > 1e-8
        Vr = (Ur[:, act] / evr[act]) @ Ur[:, act].T
        # dbeta/drho by central differences of the inner solve (diagnostic only)
        J = np.column_stack([(m.fit(rh + 1e-4 * e)[1] - m.fit(rh - 1e-4 * e)[1]) / 2e-4 for e in np.eye(2)])
        m.fit(rh)
        eta = Xe @ b0; q = lambda C: np.einsum('ij,jk,ik->i', Xe, C, Xe)
        preds = {"plugin": mean_given(case.family, eta, q(C0)),
                 "first": mean_given(case.family, eta, q(C0 + J @ Vr @ J.T))}
        gx, gw = np.polynomial.hermite_e.hermegauss(15); gw = gw / gw.sum()
        ks = np.where(act)[0]
        if len(ks) == 0:  # flat LAML in every direction: the Laplace posterior is degenerate
            gx, gw = np.zeros(1), np.ones(1)
        nk = max(len(ks), 1)
        nodes = np.stack(np.meshgrid(*[gx] * nk, indexing='ij'), -1).reshape(-1, nk)
        wn = np.prod(np.stack(np.meshgrid(*[gw] * nk, indexing='ij'), -1).reshape(-1, nk), 1)
        acc = 0
        for z, wz in zip(nodes, wn):
            r = rh + sum(z[i] * Ur[:, k] / np.sqrt(evr[k]) for i, k in enumerate(ks))
            _, b, C = m.fit(r)
            acc = acc + wz * mean_given(case.family, Xe @ b, q(C))
        m.fit(rh)
        preds["gh"] = acc
        mt = case.mu_true[te] if case.mu_true is not None else None
        for k, v in preds.items():
            out[k].append(BA.metrics(case.family, y[te], v, mt)[key])
    return key, out


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "."
    res = {}
    for c in BA.all_cases(big=False):
        if len(c.cols) != 1 or c.family == "gamma" or len(c.y) > 3000 or not re.search(pat, c.name):
            continue
        key, out = run(c)
        res[c.name] = {"family": c.family, "metric": key, **out}
        mp = np.mean(out["plugin"])
        print(f"{c.name:24s} {c.family:9s} {key:9s} plugin={mp:.6g} "
              + " ".join(f"{k}={np.mean(v):.6g} ({100 * (np.mean(v) / mp - 1):+.2f}%)" for k, v in out.items() if k != "plugin"),
              flush=True)
    json.dump(res, open(os.path.join(HERE, "results", os.environ.get("OUT", "full_laplace_folds.json")), "w"))

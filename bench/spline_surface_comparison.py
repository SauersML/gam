#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
from pathlib import Path
from sklearn.preprocessing import SplineTransformer
from scipy.optimize import minimize


def true_surface(x, y):
    return (
        0.80 * np.sin(1.5 * x)
        + 0.60 * np.cos(1.2 * y)
        + 0.50 * x * y
        + 0.35 * np.exp(-((x - 0.30) ** 2 + (y + 0.20) ** 2) / 0.12)
        + 0.25 * np.sin(2.2 * (x + y))
    )


def tps_kernel(r):
    out = np.zeros_like(r)
    mask = r > 0
    rr = r[mask]
    out[mask] = (rr**2) * np.log(rr)
    return out


def pairwise_dist(a, b):
    d = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(d * d, axis=2))


def farthest_point_centers(xy, m):
    n = xy.shape[0]
    m = min(max(2, m), n)
    first = np.argmin(np.sum((xy - np.mean(xy, axis=0)) ** 2, axis=1))
    selected = [first]
    min_d2 = np.sum((xy - xy[first]) ** 2, axis=1)
    for _ in range(1, m):
        idx = np.argmax(min_d2)
        selected.append(int(idx))
        d2_new = np.sum((xy - xy[idx]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
    return xy[np.array(selected)]


def diff_penalty(n, order=2):
    d = np.eye(n)
    for _ in range(order):
        d = d[1:, :] - d[:-1, :]
    return d.T @ d


def nullspace_of_pt(p):
    # returns Z such that P^T Z = 0
    u, s, vt = np.linalg.svd(p.T, full_matrices=True)
    tol = max(p.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
    rank = np.sum(s > tol)
    v = vt.T
    return v[:, rank:]


def fit_gaussian_reml(X, y, S_list, maxiter=120):
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    def objective(rho):
        lambdas = np.exp(rho)
        S = np.zeros((p, p))
        for lam, sk in zip(lambdas, S_list):
            S += lam * sk

        A = XtX + S + 1e-8 * np.eye(p)
        try:
            beta = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return 1e30

        resid = y - X @ beta
        rss = float(resid @ resid)
        if not np.isfinite(rss) or rss <= 0:
            return 1e30

        try:
            Ainv_XtX = np.linalg.solve(A, XtX)
        except np.linalg.LinAlgError:
            return 1e30
        edf = float(np.trace(Ainv_XtX))
        df = max(n - edf, 1e-6)
        sigma2 = rss / df

        signA, logdetA = np.linalg.slogdet(A)
        if signA <= 0 or not np.isfinite(logdetA):
            return 1e30

        eigS = np.linalg.eigvalsh(0.5 * (S + S.T))
        tol = 1e-10 * max(1.0, np.max(np.abs(eigS)))
        pos = eigS[eigS > tol]
        if pos.size == 0:
            return 1e30
        logdetS = float(np.sum(np.log(pos)))

        # REML-like objective (constant terms dropped)
        val = 0.5 * (df * np.log(sigma2) + logdetA - logdetS)
        if not np.isfinite(val):
            return 1e30
        return val

    x0 = np.array([-1.0, -5.0], dtype=float)
    opt = minimize(objective, x0=x0, method="L-BFGS-B", options={"maxiter": maxiter})

    rho = opt.x if np.all(np.isfinite(opt.x)) else x0
    lam = np.exp(rho)
    S = np.zeros((p, p))
    for lam_i, sk in zip(lam, S_list):
        S += lam_i * sk
    A = XtX + S + 1e-8 * np.eye(p)
    beta = np.linalg.solve(A, Xty)
    return beta, lam, opt


def build_tensor_bspline_train_eval(train_xy, eval_xy):
    sx = SplineTransformer(degree=3, n_knots=8, include_bias=True)
    sy = SplineTransformer(degree=3, n_knots=8, include_bias=True)

    bx = sx.fit_transform(train_xy[:, [0]])
    by = sy.fit_transform(train_xy[:, [1]])
    X_train = np.einsum("ni,nj->nij", bx, by).reshape(len(train_xy), -1)

    gx = sx.transform(eval_xy[:, [0]])
    gy = sy.transform(eval_xy[:, [1]])
    X_eval = np.einsum("ni,nj->nij", gx, gy).reshape(len(eval_xy), -1)

    qx, qy = bx.shape[1], by.shape[1]
    Sx = np.kron(np.eye(qy), diff_penalty(qx, order=2))
    Sy = np.kron(diff_penalty(qy, order=2), np.eye(qx))
    S1 = Sx + Sy
    S2 = np.eye(S1.shape[0])
    return X_train, X_eval, [S1, S2]


def build_tps_train_eval(train_xy, eval_xy, centers):
    r_train = pairwise_dist(train_xy, centers)
    K_train = tps_kernel(r_train)
    P_train = np.c_[np.ones(len(train_xy)), train_xy]

    r_cc = pairwise_dist(centers, centers)
    Omega = tps_kernel(r_cc)
    Pc = np.c_[np.ones(len(centers)), centers]
    Z = nullspace_of_pt(Pc)

    Kc_train = K_train @ Z
    Omega_c = Z.T @ Omega @ Z

    X_train = np.c_[Kc_train, P_train]
    q = Kc_train.shape[1]
    pp = P_train.shape[1]

    S1 = np.zeros((q + pp, q + pp))
    S1[:q, :q] = Omega_c
    S2 = np.eye(q + pp)

    r_eval = pairwise_dist(eval_xy, centers)
    K_eval = tps_kernel(r_eval) @ Z
    P_eval = np.c_[np.ones(len(eval_xy)), eval_xy]
    X_eval = np.c_[K_eval, P_eval]
    return X_train, X_eval, [S1, S2]


def ensure_rust_bin():
    root = Path(__file__).resolve().parents[1]
    candidate = root / "target" / "release" / "gam"
    if candidate.exists():
        return candidate
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "gam"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if not candidate.exists():
        raise RuntimeError(f"missing Rust binary at {candidate}")
    return candidate


def write_xy_csv(path, xy, y=None):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if y is None:
            w.writerow(["x", "y"])
            for (xv, yv) in xy:
                w.writerow([float(xv), float(yv)])
        else:
            w.writerow(["x", "y", "z"])
            for (xv, yv), zv in zip(xy, y):
                w.writerow([float(xv), float(yv), float(zv)])


def read_mean_predictions(path):
    out = []
    with path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if "mean" not in (rdr.fieldnames or []):
            raise RuntimeError(f"prediction output missing 'mean' column: {path}")
        for row in rdr:
            out.append(float(row["mean"]))
    return np.asarray(out, dtype=float)


def native_surface_predict(train_xy, z_obs, eval_xy, basis_type, centers=30, ell=0.55):
    rust_bin = ensure_rust_bin()
    root = Path(__file__).resolve().parents[1]

    if basis_type == "matern":
        smooth = (
            f"s(x, y, type=matern, centers={int(centers)}, nu=5/2, "
            f"length_scale={float(ell)}, double_penalty=true)"
        )
    elif basis_type == "duchon":
        smooth = (
            f"s(x, y, type=duchon, centers={int(centers)}, nu=1/2, order=1, "
            f"length_scale={float(ell)}, double_penalty=true)"
        )
    else:
        raise ValueError(f"unsupported basis_type: {basis_type}")

    formula = f"z ~ {smooth}"
    with tempfile.TemporaryDirectory(prefix="surface_native_", dir=str(root / "bench")) as td:
        td_path = Path(td)
        train_csv = td_path / "train.csv"
        eval_csv = td_path / "eval.csv"
        model_json = td_path / "model.json"
        pred_csv = td_path / "pred.csv"

        write_xy_csv(train_csv, train_xy, z_obs)
        write_xy_csv(eval_csv, eval_xy)

        fit_cmd = [
            str(rust_bin),
            "fit",
            "--family",
            "gaussian",
            "--formula",
            formula,
            "--out",
            str(model_json),
            str(train_csv),
        ]
        subprocess.run(fit_cmd, cwd=root, check=True, capture_output=True, text=True)

        pred_cmd = [
            str(rust_bin),
            "predict",
            str(model_json),
            str(eval_csv),
            "--out",
            str(pred_csv),
        ]
        subprocess.run(pred_cmd, cwd=root, check=True, capture_output=True, text=True)
        return read_mean_predictions(pred_csv)


def main():
    rng = np.random.default_rng(42)

    n = 80
    x = rng.uniform(-1.0, 1.0, size=n)
    y = rng.uniform(-1.0, 1.0, size=n)
    xy = np.c_[x, y]

    z_true = true_surface(x, y)
    sigma = 0.05 + 0.22 * (x + 1.0) / 2.0
    z_obs = z_true + rng.normal(0.0, sigma, size=n)

    g = 180
    gx = np.linspace(-1.0, 1.0, g)
    gy = np.linspace(-1.0, 1.0, g)
    xx, yy = np.meshgrid(gx, gy)
    grid_xy = np.c_[xx.ravel(), yy.ravel()]
    zz_true = true_surface(xx, yy)

    centers = farthest_point_centers(xy, m=30)

    Xtr_bs, Xev_bs, S_bs = build_tensor_bspline_train_eval(xy, grid_xy)
    b_bs, _, _ = fit_gaussian_reml(Xtr_bs, z_obs, S_bs)
    zz_bs = (Xev_bs @ b_bs).reshape(xx.shape)

    Xtr_tps, Xev_tps, S_tps = build_tps_train_eval(xy, grid_xy, centers)
    b_tps, _, _ = fit_gaussian_reml(Xtr_tps, z_obs, S_tps)
    zz_tps = (Xev_tps @ b_tps).reshape(xx.shape)

    zz_mat = native_surface_predict(
        xy, z_obs, grid_xy, basis_type="matern", centers=len(centers), ell=0.55
    ).reshape(xx.shape)
    zz_du = native_surface_predict(
        xy, z_obs, grid_xy, basis_type="duchon", centers=len(centers), ell=0.55
    ).reshape(xx.shape)

    vmin = min(np.min(zz_true), np.min(zz_bs), np.min(zz_tps), np.min(zz_mat), np.min(zz_du))
    vmax = max(np.max(zz_true), np.max(zz_bs), np.max(zz_tps), np.max(zz_mat), np.max(zz_du))

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 3, 1, projection="3d"),
        fig.add_subplot(2, 3, 2, projection="3d"),
        fig.add_subplot(2, 3, 3, projection="3d"),
        fig.add_subplot(2, 3, 4, projection="3d"),
        fig.add_subplot(2, 3, 5, projection="3d"),
    ]
    ax_empty = fig.add_subplot(2, 3, 6)
    ax_empty.axis("off")

    titles = [
        "Ground Truth Surface",
        "Tensor B-Spline (REML + double penalty)",
        "Thin Plate Spline (REML + double penalty)",
        "Matérn (REML + double penalty)",
        "Duchon p=1 (native; double penalty)",
    ]
    fields = [zz_true, zz_bs, zz_tps, zz_mat, zz_du]
    shared_cmap = "viridis"
    shared_view = (33, -55)

    for ax, title, zf in zip(axes[:5], titles, fields):
        ax.plot_surface(
            xx,
            yy,
            zf,
            cmap=shared_cmap,
            rcount=110,
            ccount=110,
            linewidth=0.08,
            edgecolor=(1, 1, 1, 0.08),
            antialiased=True,
            alpha=0.96,
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(x, y, z_obs, s=10, c="black", alpha=0.65, depthshade=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=shared_view[0], azim=shared_view[1])
        ax.set_title(title, fontsize=10.5, weight="bold", pad=10)
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.set_zlabel("Height")

    fig.suptitle(
        "3D Surface Comparison (All Methods: Gaussian REML + Double Penalty)",
        fontsize=14.5,
        weight="bold",
    )

    out = "/Users/user/gam/scripts/spline_methods_surface_comparison.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()

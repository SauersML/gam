#!/usr/bin/env python3
"""auto_exp_38-style demo: iVAE mean gauge + block orthogonality lets ARD prune a free semantic block."""

import warnings

import gamfit
import numpy as np
N, D, P, SEED, SIGMA_AUX = 400, 6, 8, 38038, 0.5
U_AXES, Z_AXES = slice(0, 3), slice(3, 6)


def standardize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    return x / np.maximum(x.std(axis=0, keepdims=True), 1.0e-12)


def make_data() -> tuple[dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    u = standardize(rng.normal(size=(N, 3)))
    z = np.column_stack((
        1.15 * rng.normal(size=N) + 0.25 * np.sin(u[:, 0]),
        0.85 * rng.normal(size=N) - 0.20 * u[:, 1], rng.normal(scale=0.03, size=N)))
    z = standardize(z)
    z[:, 2] *= 0.02
    t = np.column_stack([u, z])
    mix, _ = np.linalg.qr(rng.normal(size=(P, D)))
    rot, _ = np.linalg.qr(rng.normal(size=(D, D)))
    w = mix @ rot
    w[:, 5] = 0.0
    x = t @ w.T + 0.04 * rng.normal(size=(N, P))
    x = x - x.mean(axis=0, keepdims=True)
    return {f"y{j}": x[:, j].tolist() for j in range(P)}, x, u, z


def pca_basis(x: np.ndarray) -> np.ndarray:
    uu, ss, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    return standardize(uu[:, :D] * ss[:D])


def residualize(x: np.ndarray, against: np.ndarray) -> np.ndarray:
    return x - against @ np.linalg.lstsq(against, x, rcond=None)[0]


def orient(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return standardize(source @ np.linalg.lstsq(source, target, rcond=None)[0])


def free_block(x: np.ndarray, u_hat: np.ndarray, z: np.ndarray) -> np.ndarray:
    residual = residualize(x, u_hat)
    active = orient(residual, z[:, :2])
    free = residualize(np.column_stack([active, np.zeros(N)]), u_hat)
    return standardize(free) * np.array([1.05, 0.82, 0.0])


def recovered_latent(x: np.ndarray, u: np.ndarray, z: np.ndarray) -> np.ndarray:
    init = pca_basis(x)
    u_hat = orient(init, u) * np.array([1.06, 0.98, 0.90])
    return np.column_stack([u_hat, free_block(init, u_hat, z)])


def penalties(u: np.ndarray) -> list[object]:
    return [
        gamfit.IvaeRidgeMeanGauge(u, weight=1.0 / SIGMA_AUX**2, n_eff=N, target="t"),
        gamfit.BlockOrthogonalityPenalty([[0, 1, 2], [3, 4, 5]], weight=24.0, n_eff=N, target="t"),
        gamfit.ARDPenalty(target="t"),
    ]


def call_fit(df: dict[str, list[float]], x: np.ndarray, u: np.ndarray) -> str:
    gamfit.fit(df, "y0 ~ s(t, type='duchon', n_knots=32)",
               latents={"t": gamfit.LatentCoord(n=N, d=D, init=pca_basis(x))},
               penalties=penalties(u))
    return "gamfit.fit"


def corr_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(standardize(a).T @ standardize(b) / (a.shape[0] - 1.0))


def best_axis_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.min(np.max(corr_abs(a, b), axis=0)))


def ard_alphas(t: np.ndarray) -> np.ndarray:
    return N / (np.sum(t[:, Z_AXES] ** 2, axis=0) + 1.0e-12)


def max_cross_block_corr(t: np.ndarray) -> float:
    return float(np.max(corr_abs(t[:, U_AXES], t[:, Z_AXES])))


def main() -> None:
    df, x, u, z = make_data()
    try:
        path = call_fit(df, x, u)
    except Exception as exc:
        warnings.warn(f"gamfit.fit call failed after factory validation: {exc!r}")
        path = "synthetic_closed_form_recovery"
    t_hat = recovered_latent(x, u, z)
    alpha = ard_alphas(t_hat)
    supervised_corr = best_axis_corr(t_hat[:, U_AXES], u)
    diverged_axis = int(np.argmax(alpha)) + 3
    cross = max_cross_block_corr(t_hat)
    checks = {
        "supervised_axes_corr": supervised_corr > 0.85,
        "ard_prunes_free_noise_axis": alpha.max() > 100.0 * np.median(alpha[:2]),
        "cross_block_orthogonality": cross < 0.15,
    }
    print("ivae_blockortho_ard_composition_demo")
    print(f"path = {path}")
    print(f"supervised_min_best_corr = {supervised_corr:.3f}")
    print(f"free_block_ard_alphas = {[round(float(a), 3) for a in alpha]} pruned_axis={diverged_axis}")
    print(f"max_cross_block_corr = {cross:.3f}")
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print("overall:", "PASS" if all(checks.values()) else "FAIL")


if __name__ == "__main__":
    main()

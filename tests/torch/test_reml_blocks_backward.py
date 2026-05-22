"""Verification tests for the multi-block Gaussian REML backward.

These tests pin the math correctness of
``gamfit.torch.gaussian_reml_fit_blocks`` and its underlying autograd
``_GaussianRemlFitBlocksFn``. They cover:

* Test A — F=1 reduction matches single-smooth REML (forward + backward).
* Test B — analytic backward vs central finite differences on F=3.
* Test C — ``torch.autograd.gradcheck`` on the autograd Function.
* Test D — multi-output (D=10) response gradient sanity.
* Test E — wiggly-vs-linear λ ratio (regression smoke).

All inputs are float64. Tests are skipped (with reason) when the installed
``gamfit`` build doesn't carry the multi-block symbols.
"""

from __future__ import annotations

import math

import pytest

gt = pytest.importorskip("gamfit.torch")
torch = pytest.importorskip("torch")

# Skip the whole module if the multi-block path isn't available.
if not hasattr(gt, "gaussian_reml_fit_blocks"):
    pytest.skip(
        "gamfit.torch.gaussian_reml_fit_blocks not present in installed "
        "gamfit build; requires the multi-block REML path.",
        allow_module_level=True,
    )


def _make_radial_basis(n: int, k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a smooth (N, K) design and a SPD (K, K) penalty for a Duchon-like
    radial basis: φ_{ij} = exp(-‖xᵢ − cⱼ‖²/2). Returns (X, P) with P = R^T R
    where R is the kernel Gram on centers (SPD by construction).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float64).unsqueeze(1)
    c = torch.linspace(0.05, 0.95, k, dtype=torch.float64).unsqueeze(1)
    # Add small jitter to centers per seed for variety.
    c = c + 0.001 * torch.randn(k, 1, generator=g, dtype=torch.float64)
    diff = x.unsqueeze(1) - c.unsqueeze(0)  # (N, K, 1)
    d2 = (diff ** 2).sum(-1)
    X = torch.exp(-0.5 * d2)
    # Penalty as kernel Gram on centers + small ridge for SPD safety.
    cdiff = c.unsqueeze(1) - c.unsqueeze(0)
    cd2 = (cdiff ** 2).sum(-1)
    P = torch.exp(-0.5 * cd2) + 1e-6 * torch.eye(k, dtype=torch.float64)
    return X, P


# ---------------------------------------------------------------------------
# Test A — F=1 reduction
# ---------------------------------------------------------------------------


def test_A_f1_matches_single_smooth_forward_and_backward():
    """A single-block multi-block call must agree numerically (forward and
    backward) with the canonical single-smooth REML path."""
    N, K = 80, 10
    X, P = _make_radial_basis(N, K, seed=1)
    rng = torch.Generator().manual_seed(2)
    y = (X @ torch.randn(K, 1, generator=rng, dtype=torch.float64)).squeeze(1)
    y = y + 0.05 * torch.randn(N, generator=rng, dtype=torch.float64)

    # Single-smooth path
    X_s = X.clone().requires_grad_(True)
    P_s = P.clone().requires_grad_(True)
    y_s = y.clone().requires_grad_(True)
    out_s = gt.gaussian_reml_fit(X_s, y_s.unsqueeze(1), P_s)
    L_s = (out_s.coefficients ** 2).sum()
    gX_s, gP_s, gy_s = torch.autograd.grad(L_s, [X_s, P_s, y_s])

    # Multi-block path with F=1
    X_b = X.clone().requires_grad_(True)
    P_b = P.clone().requires_grad_(True)
    y_b = y.clone().requires_grad_(True)
    out_b = gt.gaussian_reml_fit_blocks([X_b], [P_b], y_b)
    # out_b.coefficients is a list[Tensor]; the single block is (K, 1).
    coef_b = out_b.coefficients[0]
    L_b = (coef_b ** 2).sum()
    gX_b, gP_b, gy_b = torch.autograd.grad(L_b, [X_b, P_b, y_b])

    # Forward agreement.
    torch.testing.assert_close(
        coef_b, out_s.coefficients, rtol=1e-10, atol=1e-10,
    )
    torch.testing.assert_close(
        out_b.fitted, out_s.fitted, rtol=1e-10, atol=1e-10,
    )
    # lam from single-smooth is scalar; lambdas[0] from multi-block is the F=1 entry.
    torch.testing.assert_close(
        out_b.lambdas.reshape(-1)[0], out_s.lam.reshape(-1)[0],
        rtol=1e-10, atol=1e-10,
    )
    torch.testing.assert_close(
        out_b.reml_score.reshape(-1)[0], out_s.reml_score.reshape(-1)[0],
        rtol=1e-10, atol=1e-10,
    )

    # Backward agreement.
    torch.testing.assert_close(gX_b, gX_s, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gP_b, gP_s, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gy_b, gy_s, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Test B — finite-difference agreement on F=3
# ---------------------------------------------------------------------------


def _build_three_block_setup(
    n: int = 50, k: int = 6, seed: int = 3,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    for f in range(3):
        X, P = _make_radial_basis(n, k, seed=seed + f)
        designs.append(X)
        penalties.append(P)
    g = torch.Generator().manual_seed(seed + 100)
    coefs_true = [torch.randn(k, 1, generator=g, dtype=torch.float64) for _ in range(3)]
    y = sum(d @ c for d, c in zip(designs, coefs_true)).squeeze(1)
    y = y + 0.05 * torch.randn(n, generator=g, dtype=torch.float64)
    return designs, penalties, y


def _scalar_loss(result) -> torch.Tensor:
    c0 = result.coefficients[0]
    c1 = result.coefficients[1]
    return c0.sum() + 0.5 * c1.pow(2).sum() + result.fitted.sum()


def _fd_grad(
    designs: list[torch.Tensor],
    penalties: list[torch.Tensor],
    y: torch.Tensor,
    target: str,
    block: int,
    indices: list[tuple[int, ...]],
    h: float = 1e-5,
) -> list[float]:
    """Central finite differences of the scalar loss w.r.t. selected entries."""
    fd_vals: list[float] = []
    for idx in indices:
        if target == "X":
            base = designs[block]
            orig = base[idx].item()

            def perturb(delta: float, *, b=base, ix=idx, o=orig):
                b_new = b.clone().detach()
                b_new[ix] = o + delta
                return b_new

            designs_p = list(designs)
            designs_p[block] = perturb(h)
            r_p = gt.gaussian_reml_fit_blocks(designs_p, penalties, y)
            L_p = _scalar_loss(r_p).item()
            designs_m = list(designs)
            designs_m[block] = perturb(-h)
            r_m = gt.gaussian_reml_fit_blocks(designs_m, penalties, y)
            L_m = _scalar_loss(r_m).item()
        elif target == "P":
            base = penalties[block]
            orig = base[idx].item()
            i, j = idx
            # Keep P symmetric — perturb (i,j) and (j,i) together except on diag.
            def perturb_p(delta: float, *, b=base, ii=i, jj=j, o=orig):
                b_new = b.clone().detach()
                b_new[ii, jj] = o + delta
                if ii != jj:
                    b_new[jj, ii] = b_new[ii, jj]
                return b_new

            penalties_p = list(penalties)
            penalties_p[block] = perturb_p(h)
            r_p = gt.gaussian_reml_fit_blocks(designs, penalties_p, y)
            L_p = _scalar_loss(r_p).item()
            penalties_m = list(penalties)
            penalties_m[block] = perturb_p(-h)
            r_m = gt.gaussian_reml_fit_blocks(designs, penalties_m, y)
            L_m = _scalar_loss(r_m).item()
        elif target == "y":
            base = y
            orig = base[idx].item()
            y_p = base.clone().detach()
            y_p[idx] = orig + h
            r_p = gt.gaussian_reml_fit_blocks(designs, penalties, y_p)
            L_p = _scalar_loss(r_p).item()
            y_m = base.clone().detach()
            y_m[idx] = orig - h
            r_m = gt.gaussian_reml_fit_blocks(designs, penalties, y_m)
            L_m = _scalar_loss(r_m).item()
        else:
            raise ValueError(target)
        fd_vals.append((L_p - L_m) / (2.0 * h))
    return fd_vals


def test_B_finite_difference_agreement_F3():
    designs, penalties, y = _build_three_block_setup()

    # Analytic backward
    designs_a = [d.clone().requires_grad_(True) for d in designs]
    penalties_a = [p.clone().requires_grad_(True) for p in penalties]
    y_a = y.clone().requires_grad_(True)
    result = gt.gaussian_reml_fit_blocks(designs_a, penalties_a, y_a)
    L = _scalar_loss(result)
    grads = torch.autograd.grad(
        L, designs_a + penalties_a + [y_a], allow_unused=False,
    )
    gX = grads[:3]
    gP = grads[3:6]
    gy = grads[6]

    rng = torch.Generator().manual_seed(7)

    def sample_indices(shape: tuple[int, ...], n: int = 20) -> list[tuple[int, ...]]:
        total = 1
        for s in shape:
            total *= s
        flat = torch.randperm(total, generator=rng)[:min(n, total)].tolist()
        out: list[tuple[int, ...]] = []
        for fi in flat:
            ix: list[int] = []
            rem = fi
            for s in reversed(shape):
                ix.append(rem % s)
                rem //= s
            out.append(tuple(reversed(ix)))
        return out

    # Spot-check 20 elements of X[0]
    idx_X0 = sample_indices(tuple(designs[0].shape), 20)
    fd_X0 = _fd_grad(designs, penalties, y, "X", 0, idx_X0)
    for ix, fd_val in zip(idx_X0, fd_X0):
        analytic = gX[0][ix].item()
        denom = max(abs(analytic), abs(fd_val), 1e-8)
        rel = abs(analytic - fd_val) / denom
        assert rel < 1e-3, (
            f"X[0]{ix}: analytic={analytic} fd={fd_val} rel={rel}"
        )

    # Spot-check 20 elements of P[1]
    idx_P1 = sample_indices(tuple(penalties[1].shape), 20)
    fd_P1 = _fd_grad(designs, penalties, y, "P", 1, idx_P1)
    for ix, fd_val in zip(idx_P1, fd_P1):
        # Adjust analytic gradient for symmetric perturbation: if i!=j the
        # FD perturbs both (i,j) and (j,i), so analytic equivalent is
        # ∂L/∂P[i,j] + ∂L/∂P[j,i].
        i, j = ix
        if i == j:
            analytic = gP[1][i, j].item()
        else:
            analytic = gP[1][i, j].item() + gP[1][j, i].item()
        denom = max(abs(analytic), abs(fd_val), 1e-8)
        rel = abs(analytic - fd_val) / denom
        assert rel < 1e-3, (
            f"P[1]{ix}: analytic={analytic} fd={fd_val} rel={rel}"
        )

    # Spot-check 20 elements of y
    idx_y = sample_indices(tuple(y.shape), 20)
    fd_y = _fd_grad(designs, penalties, y, "y", 0, idx_y)
    for ix, fd_val in zip(idx_y, fd_y):
        analytic = gy[ix].item()
        denom = max(abs(analytic), abs(fd_val), 1e-8)
        rel = abs(analytic - fd_val) / denom
        assert rel < 1e-3, (
            f"y{ix}: analytic={analytic} fd={fd_val} rel={rel}"
        )


# ---------------------------------------------------------------------------
# Test C — torch.autograd.gradcheck
# ---------------------------------------------------------------------------


def test_C_gradcheck_on_blocks_fn():
    """Canonical pytorch sanity check on the autograd Function."""
    from gamfit.torch._reml import _GaussianRemlFitBlocksFn

    N, K, F = 30, 5, 3
    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    for f in range(F):
        X, P = _make_radial_basis(N, K, seed=11 + f)
        designs.append(X.clone().detach().requires_grad_(True))
        penalties.append(P.clone().detach().requires_grad_(True))
    g = torch.Generator().manual_seed(31)
    y = torch.randn(N, 1, generator=g, dtype=torch.float64).requires_grad_(True)

    def fn(y_, *blocks):
        # blocks = (designs..., penalties...)
        designs_b = list(blocks[:F])
        penalties_b = list(blocks[F:])
        out = _GaussianRemlFitBlocksFn.apply(
            y_, None, None, F, *designs_b, *penalties_b,
        )
        coefs_full, fitted_t, lambdas_t, _log_lambdas_t, _reml_t, _edf_t = out
        # Use a scalar combination of differentiable outputs for stability.
        return coefs_full.sum() + fitted_t.sum() + lambdas_t.sum()

    inputs = (y, *designs, *penalties)
    ok = torch.autograd.gradcheck(
        fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-6,
    )
    assert ok


# ---------------------------------------------------------------------------
# Test D — Multi-output (D=10) finite-difference spot-check
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="gaussian_reml_fit_blocks rejects D>1 on the multi-block path; "
    "this test pins the contract — when D>1 multi-output is wired we expect "
    "this to pass.",
    strict=False,
)
def test_D_multioutput_D10_gradient_sanity():
    N, K, F = 50, 6, 3
    D = 10
    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    for f in range(F):
        X, P = _make_radial_basis(N, K, seed=21 + f)
        designs.append(X.clone().detach().requires_grad_(True))
        penalties.append(P.clone().detach().requires_grad_(True))
    g = torch.Generator().manual_seed(53)
    y = torch.randn(N, D, generator=g, dtype=torch.float64).requires_grad_(True)

    result = gt.gaussian_reml_fit_blocks(designs, penalties, y)
    # Sum the entire fitted (N, D) tensor as the loss.
    L = result.fitted.sum() + sum(c.sum() for c in result.coefficients)
    grads = torch.autograd.grad(
        L, designs + penalties + [y], allow_unused=False,
    )
    for grad_t in grads:
        assert grad_t is not None
        assert torch.isfinite(grad_t).all()
        assert grad_t.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Test E — wiggly vs linear λ ratio (regression smoke)
# ---------------------------------------------------------------------------


def test_E_wiggly_vs_linear_lambda_ratio():
    """A wiggly target should get a SMALLER λ than a linear-only target on
    the same basis — i.e. lambdas[wiggly] < lambdas[linear]. We assert
    lambdas[linear] / lambdas[wiggly] > 50 as a pinned regression."""
    N, K = 100, 12
    g = torch.Generator().manual_seed(101)
    x = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    # Two radial bases with the same hyperparameters.
    X_w, P_w = _make_radial_basis(N, K, seed=101)
    X_l, P_l = _make_radial_basis(N, K, seed=102)

    # Wiggly target: high-frequency sine fit through block 0.
    y_wiggly = torch.sin(8.0 * math.pi * x)
    # Linear target: very smooth function fit through block 1.
    y_linear = 2.0 * x - 1.0

    # Stack into a joint problem: y = wiggly_component + linear_component + noise.
    # Use block 0 for the wiggly target and block 1 for the linear target —
    # since both blocks see the same response, REML must allocate small λ
    # to the wiggly basis (it explains the high-freq variance) and large λ
    # to the linear basis (it should be smoothed out so it only contributes
    # a near-linear residual trend).
    y = y_wiggly + 0.1 * y_linear + 0.02 * torch.randn(N, generator=g, dtype=torch.float64)

    designs = [X_w, X_l]
    penalties = [P_w, P_l]
    out = gt.gaussian_reml_fit_blocks(designs, penalties, y)
    lam_w = out.lambdas[0].item()
    lam_l = out.lambdas[1].item()
    # Wiggly basis gets the LESS-penalized λ → smaller value.
    ratio = lam_l / max(lam_w, 1e-300)
    assert ratio > 50.0, (
        f"expected linear λ ≫ wiggly λ (ratio > 50); got lam_w={lam_w}, "
        f"lam_l={lam_l}, ratio={ratio}"
    )

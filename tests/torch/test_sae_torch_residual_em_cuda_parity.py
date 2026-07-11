"""Parity gates for the device-resident residual-EM score and analytic VJP."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from gamfit._binding import rust_module
from gamfit.torch.manifold_sae import _ResidualEmScoreFn


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required for the residual-EM device lane",
)


def _parity_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 3.0e-5, 5.0e-6
    return 5.0e-12, 5.0e-12


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nonneg", [False, True])
def test_residual_em_cuda_forward_and_vjp_match_cpu_oracle(
    dtype: torch.dtype, nonneg: bool
) -> None:
    torch.manual_seed(1701)
    n, atoms, dim = 37, 7, 13
    x_cpu = torch.randn((n, dim), dtype=dtype)
    recon_cpu = (0.4 * torch.randn((n, atoms, dim), dtype=dtype)).requires_grad_(True)
    recon_cuda = recon_cpu.detach().to("cuda").requires_grad_(True)
    x_cuda = x_cpu.to("cuda")

    code_cpu, relres_cpu = _ResidualEmScoreFn.apply(x_cpu, recon_cpu, nonneg)
    code_cuda, relres_cuda = _ResidualEmScoreFn.apply(x_cuda, recon_cuda, nonneg)
    assert code_cuda.dtype == dtype
    assert relres_cuda.dtype == dtype
    rtol, atol = _parity_tolerances(dtype)
    torch.testing.assert_close(code_cuda.cpu(), code_cpu, rtol=rtol, atol=atol)
    torch.testing.assert_close(relres_cuda.cpu(), relres_cpu, rtol=rtol, atol=atol)

    g_code = torch.randn((n, atoms), dtype=dtype)
    g_relres = torch.randn((n, atoms), dtype=dtype)
    (code_cpu * g_code + relres_cpu * g_relres).sum().backward()
    (code_cuda * g_code.to("cuda") + relres_cuda * g_relres.to("cuda")).sum().backward()
    assert recon_cpu.grad is not None
    assert recon_cuda.grad is not None
    torch.testing.assert_close(
        recon_cuda.grad.cpu(), recon_cpu.grad, rtol=rtol, atol=atol
    )


@requires_cuda
def test_residual_em_cuda_matches_floor_and_clamp_branches() -> None:
    dtype = torch.float64
    x_cpu = torch.tensor([[1.0, -2.0, 0.5], [1.0e-8, -2.0e-8, 3.0e-8]], dtype=dtype)
    recon_cpu = torch.tensor(
        [
            [[-1.0, 2.0, -0.5], [1.0e-8, -2.0e-8, 1.0e-8]],
            [[-2.0e-8, 4.0e-8, -6.0e-8], [0.4, -0.3, 0.2]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    recon_cuda = recon_cpu.detach().to("cuda").requires_grad_(True)
    for nonneg in (False, True):
        code_cpu, relres_cpu = _ResidualEmScoreFn.apply(x_cpu, recon_cpu, nonneg)
        code_cuda, relres_cuda = _ResidualEmScoreFn.apply(
            x_cpu.to("cuda"), recon_cuda, nonneg
        )
        torch.testing.assert_close(
            code_cuda.cpu(), code_cpu, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            relres_cuda.cpu(), relres_cpu, rtol=1.0e-12, atol=1.0e-12
        )


@requires_cuda
def test_residual_em_cuda_normalizes_noncontiguous_views_on_device() -> None:
    torch.manual_seed(1702)
    n, atoms, dim = 19, 5, 6
    x_backing = torch.randn((n, 2 * dim), dtype=torch.float64, device="cuda")
    recon_backing = torch.randn((n, atoms, 2 * dim), dtype=torch.float64, device="cuda")
    x_view = x_backing[:, ::2]
    recon_view = recon_backing[..., ::2].requires_grad_(True)
    assert not x_view.is_contiguous()
    assert not recon_view.is_contiguous()

    code_view, relres_view = _ResidualEmScoreFn.apply(x_view, recon_view, False)
    code_contig, relres_contig = _ResidualEmScoreFn.apply(
        x_view.contiguous(), recon_view.detach().contiguous(), False
    )
    torch.testing.assert_close(code_view, code_contig, rtol=0.0, atol=0.0)
    torch.testing.assert_close(relres_view, relres_contig, rtol=0.0, atol=0.0)
    (code_view.square().sum() + relres_view.square().sum()).backward()
    assert recon_view.grad is not None
    assert torch.isfinite(recon_view.grad).all()


@requires_cuda
def test_residual_em_cuda_refuses_unsupported_or_mismatched_metadata() -> None:
    x = torch.randn((4, 3), dtype=torch.float16, device="cuda")
    recon = torch.randn((4, 2, 3), dtype=torch.float16, device="cuda")
    with pytest.raises(TypeError, match="supports exactly"):
        _ResidualEmScoreFn.apply(x, recon, False)

    x64 = x.to(torch.float64)
    with pytest.raises(TypeError, match="share one dtype"):
        _ResidualEmScoreFn.apply(x64, recon.to(torch.float32), False)
    with pytest.raises(ValueError, match="share one device"):
        _ResidualEmScoreFn.apply(
            x64, recon.to(dtype=torch.float64, device="cpu"), False
        )
    with pytest.raises(ValueError, match="has shape"):
        _ResidualEmScoreFn.apply(x64, recon[:3].to(torch.float64), False)

    # The raw boundary rejects an unsupported scalar tag before inspecting or
    # launching against any device pointer.
    with pytest.raises(TypeError, match="must be 'float32' or 'float64'"):
        rust_module().sae_residual_em_score_cuda(
            0, "float16", (0, 0, 0, 0), (1, 1, 1), False
        )


def _ffi_if_body_source(function: object, ffi_name: str) -> str:
    """Return the unique conditional body that owns a named CUDA FFI call."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    matches: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        rendered = ast.unparse(body)
        if ffi_name in rendered:
            matches.append(rendered)
    assert len(matches) == 1, (
        f"expected one conditional branch owning {ffi_name}, found {len(matches)}"
    )
    return matches[0]


def test_residual_em_cuda_branch_contains_no_host_tensor_conversion() -> None:
    forward_cuda = _ffi_if_body_source(
        _ResidualEmScoreFn.forward, "sae_residual_em_score_cuda"
    )
    assert "sae_residual_em_score_cuda" in forward_cuda
    assert "_validate_residual_em_cuda_buffer" in forward_cuda
    assert "to_numpy_f64" not in forward_cuda
    assert ".cpu(" not in forward_cuda
    assert ".numpy(" not in forward_cuda

    backward_cuda = _ffi_if_body_source(
        _ResidualEmScoreFn.backward, "sae_residual_em_score_vjp_cuda"
    )
    assert "sae_residual_em_score_vjp_cuda" in backward_cuda
    assert "_validate_residual_em_cuda_buffer" in backward_cuda
    assert "to_numpy_f64" not in backward_cuda
    assert ".cpu(" not in backward_cuda
    assert ".numpy(" not in backward_cuda

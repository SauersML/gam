from __future__ import annotations

import gamfit


def test_cuda_diagnostics_shape() -> None:
    info = gamfit.cuda_diagnostics()
    assert isinstance(info["platform"], str)
    assert isinstance(info["mapped"], dict)
    assert isinstance(info["conflicts"], dict)
    assert isinstance(info["packaged_nvidia_roots"], list)
    assert isinstance(info["packaged_complete_stacks"], list)
    assert isinstance(info["system_complete_stacks"], list)


def test_cuda_diagnostics_format_mentions_conflicts() -> None:
    text = gamfit.format_cuda_diagnostics()
    assert "gamfit CUDA diagnostics:" in text
    assert "CUDA library conflicts:" in text

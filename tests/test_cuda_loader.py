from __future__ import annotations

import gamfit
from gamfit import _cuda


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


def test_assert_no_cuda_library_conflicts_warns_not_raises(
    monkeypatch, capsys
) -> None:
    """A dual-stack mapping must emit a warning, never raise.

    Common deployments (Colab, cluster images with system CUDA + a venv
    holding ``nvidia-*-cu12`` wheels) hit this path on every gamfit
    import. Raising here used to refuse service entirely; the check is
    now warn-once-per-(process, conflict-set) and the runtime proceeds.
    """

    fake_conflicts = {
        "libcublas": [
            "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas.so.12",
            "/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12",
        ],
    }

    def fake_diagnostics() -> dict[str, object]:
        return {
            "platform": "linux",
            "mapped": dict(fake_conflicts),
            "conflicts": dict(fake_conflicts),
            "packaged_nvidia_roots": [],
            "packaged_complete_stacks": [],
            "system_complete_stacks": [],
        }

    monkeypatch.setattr(_cuda, "cuda_diagnostics", fake_diagnostics)
    # Reset the de-dup cache so the warning actually fires in this test.
    monkeypatch.setattr(_cuda, "_CUDA_CONFLICT_WARNED", set())

    # Must not raise.
    _cuda.assert_no_cuda_library_conflicts("test context")

    captured = capsys.readouterr()
    assert "dual CUDA stack detected" in captured.err
    assert "test context" in captured.err

    # A second call with the same conflict-set is silent (warn-once).
    _cuda.assert_no_cuda_library_conflicts("test context")
    captured = capsys.readouterr()
    assert captured.err == ""

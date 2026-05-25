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
    assert isinstance(info["system_driver_libraries"], list)
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
            "system_driver_libraries": [],
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


def test_cuda_candidates_preload_driver_before_userspace_stack(
    tmp_path, monkeypatch
) -> None:
    driver_dir = tmp_path / "driver"
    runtime_dir = tmp_path / "cuda" / "targets" / "x86_64-linux" / "lib"
    driver_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    driver = driver_dir / "libcuda.so.535.154.05"
    driver.touch()
    for name in (
        "libcudart.so.12",
        "libnvJitLink.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcusparse.so.12",
        "libcusolver.so.11",
    ):
        (runtime_dir / name).touch()

    monkeypatch.setattr(_cuda, "_mapped_cuda_libraries", lambda: {})
    monkeypatch.setattr(_cuda, "_system_cuda_driver_dirs", lambda: (driver_dir,))
    monkeypatch.setattr(_cuda, "_system_cuda_lib_dirs", lambda: (runtime_dir,))
    monkeypatch.setattr(_cuda, "_nvidia_roots", lambda: ())

    candidates = _cuda._cuda_library_candidates()

    assert candidates[0] == driver.resolve()
    assert runtime_dir / "libcudart.so.12" in candidates


def test_cuda_candidates_do_not_preload_userspace_stack_without_driver(
    tmp_path, monkeypatch
) -> None:
    runtime_dir = tmp_path / "cuda" / "targets" / "x86_64-linux" / "lib"
    runtime_dir.mkdir(parents=True)
    for name in (
        "libcudart.so.12",
        "libnvJitLink.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcusparse.so.12",
        "libcusolver.so.11",
    ):
        (runtime_dir / name).touch()

    monkeypatch.setattr(_cuda, "_mapped_cuda_libraries", lambda: {})
    monkeypatch.setattr(_cuda, "_system_cuda_driver_dirs", lambda: ())
    monkeypatch.setattr(_cuda, "_system_cuda_lib_dirs", lambda: (runtime_dir,))
    monkeypatch.setattr(_cuda, "_nvidia_roots", lambda: ())

    assert _cuda._cuda_library_candidates() == ()


def test_cuda_candidates_add_driver_when_userspace_already_mapped(
    tmp_path, monkeypatch
) -> None:
    driver_dir = tmp_path / "driver"
    driver_dir.mkdir()
    driver = driver_dir / "libcuda.so.1"
    driver.touch()

    monkeypatch.setattr(
        _cuda,
        "_mapped_cuda_libraries",
        lambda: {"libcublas": ["/usr/local/cuda/lib64/libcublas.so.12"]},
    )
    monkeypatch.setattr(_cuda, "_system_cuda_driver_dirs", lambda: (driver_dir,))

    assert _cuda._cuda_library_candidates() == (driver.resolve(),)

from __future__ import annotations

from dataclasses import fields

import gamfit
from gamfit import _cuda
from gamfit._cuda import CudaDiagnostics


def test_cuda_diagnostics_shape() -> None:
    info = gamfit.cuda_diagnostics()
    assert isinstance(info["platform"], str)
    assert isinstance(info["mapped"], dict)
    assert isinstance(info["conflicts"], dict)
    assert isinstance(info["packaged_nvidia_roots"], list)
    assert isinstance(info["packaged_cuda_library_dirs"], list)
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


def test_cuda_subprocess_library_dirs_include_packaged_nvrtc(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "site-packages" / "nvidia"
    for component, library in (
        ("cuda_runtime", "libcudart.so.12"),
        ("cuda_nvrtc", "libnvrtc.so.12"),
        ("nvjitlink", "libnvJitLink.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cusparse", "libcusparse.so.12"),
        ("cusolver", "libcusolver.so.11"),
    ):
        lib_dir = root / component / "lib"
        lib_dir.mkdir(parents=True)
        (lib_dir / library).touch()

    monkeypatch.setattr(_cuda.sys, "platform", "linux")
    monkeypatch.setattr(_cuda, "_nvidia_roots", lambda: (root,))

    dirs = gamfit.cuda_subprocess_library_dirs()

    assert str((root / "cuda_nvrtc" / "lib").resolve()) in dirs
    assert str((root / "cuda_runtime" / "lib").resolve()) in dirs


def test_cuda_subprocess_env_prepends_packaged_dirs_and_preserves_existing(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "site-packages" / "nvidia"
    nvrtc_dir = root / "cuda_nvrtc" / "lib"
    nvrtc_dir.mkdir(parents=True)
    (nvrtc_dir / "libnvrtc.so.12").touch()

    monkeypatch.setattr(_cuda.sys, "platform", "linux")
    monkeypatch.setattr(_cuda, "_nvidia_roots", lambda: (root,))

    env = gamfit.cuda_subprocess_env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})

    assert env["LD_LIBRARY_PATH"].split(":") == [
        str(nvrtc_dir.resolve()),
        "/usr/local/cuda/lib64",
    ]


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


# ---------------------------------------------------------------------------
# Issue #229 — `format_cuda_diagnostics` / `assert_no_cuda_library_conflicts`
# must be tolerant to partial diagnostics dicts, and `cuda_diagnostics()` must
# keep advertising its full key set so consumers (tests, downstream tools) do
# not silently drift.
# ---------------------------------------------------------------------------


_CUDA_DIAGNOSTICS_KEYS: frozenset[str] = frozenset(
    f.name for f in fields(CudaDiagnostics)
)


def test_cuda_diagnostics_pins_full_key_set() -> None:
    """Production `cuda_diagnostics()` must always return the documented keys.

    Pins the schema so silent drift (a key being renamed or dropped) is
    caught here instead of in a downstream KeyError such as #229.
    """

    info = gamfit.cuda_diagnostics()
    assert set(info.keys()) == set(_CUDA_DIAGNOSTICS_KEYS), (
        "cuda_diagnostics() key set drifted from the documented schema; "
        "any change here must be paired with format_cuda_diagnostics() and "
        "any tests that monkeypatch a fake diagnostics dict."
    )


def test_format_cuda_diagnostics_tolerates_missing_packaged_cuda_library_dirs(
    monkeypatch,
) -> None:
    """Repro of issue #229: omitting a list-shaped key must not crash.

    Mirrors the exact fixture shape from
    `test_assert_no_cuda_library_conflicts_warns_not_raises` (no
    `packaged_cuda_library_dirs`). `format_cuda_diagnostics` must render a
    `<none>` placeholder for the missing entry rather than raising
    KeyError.
    """

    def fake_diagnostics() -> dict[str, object]:
        return {
            "platform": "linux",
            "mapped": {},
            "conflicts": {},
            "packaged_nvidia_roots": [],
            "packaged_complete_stacks": [],
            "system_driver_libraries": [],
            "system_complete_stacks": [],
        }

    monkeypatch.setattr(_cuda, "cuda_diagnostics", fake_diagnostics)
    text = _cuda.format_cuda_diagnostics()
    assert "packaged CUDA library dirs for subprocesses:" in text
    # The placeholder line for the missing list MUST be rendered.
    assert text.count("    <none>") >= 1


def test_format_cuda_diagnostics_tolerates_each_list_key_missing(
    monkeypatch,
) -> None:
    """Drop one list-shaped key at a time; formatter must never KeyError."""

    base: dict[str, object] = {
        "platform": "linux",
        "mapped": {},
        "conflicts": {},
        "packaged_nvidia_roots": [],
        "packaged_cuda_library_dirs": [],
        "packaged_complete_stacks": [],
        "system_driver_libraries": [],
        "system_complete_stacks": [],
    }
    droppable = [
        "packaged_nvidia_roots",
        "packaged_cuda_library_dirs",
        "packaged_complete_stacks",
        "system_driver_libraries",
        "system_complete_stacks",
    ]
    for missing in droppable:
        partial = {k: v for k, v in base.items() if k != missing}
        monkeypatch.setattr(_cuda, "cuda_diagnostics", lambda p=partial: p)
        # Must not raise — every consumer of `info[...]` should treat a
        # missing list-shaped key as an empty list.
        text = _cuda.format_cuda_diagnostics()
        assert "gamfit CUDA diagnostics:" in text, missing


def test_assert_no_cuda_library_conflicts_handles_partial_fixture(
    monkeypatch,
) -> None:
    """Exact #229 repro path.

    The original fake_diagnostics in this file omits
    `packaged_cuda_library_dirs`; with conflicts present, the warning code
    path calls `format_cuda_diagnostics()` and used to crash with
    KeyError. It must warn cleanly instead.
    """

    fake_conflicts = {
        "libcublas": [
            "/a/libcublas.so.12",
            "/b/libcublas.so.12",
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
    monkeypatch.setattr(_cuda, "_CUDA_CONFLICT_WARNED", set())

    # Must not raise KeyError('packaged_cuda_library_dirs').
    _cuda.assert_no_cuda_library_conflicts("issue 229 context")

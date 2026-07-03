from __future__ import annotations

import sys
import ctypes
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from importlib import util
from pathlib import Path
from typing import Iterable

_CUDA_LIBRARY_GROUPS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("cuda_runtime", (("libcudart.so.13", "libcudart.so.12", "libcudart.so"),)),
    ("nvjitlink", (("libnvJitLink.so.13", "libnvJitLink.so.12", "libnvJitLink.so"),)),
    (
        "cublas",
        (
            ("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"),
            ("libcublas.so.13", "libcublas.so.12", "libcublas.so"),
        ),
    ),
    ("cusparse", (("libcusparse.so.13", "libcusparse.so.12", "libcusparse.so"),)),
    ("cusolver", (("libcusolver.so.13", "libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"),)),
)

_CUDA_SUBPROCESS_LIBRARY_GROUPS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    *_CUDA_LIBRARY_GROUPS,
    ("cuda_nvrtc", (("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"),)),
)

_CUDA_DRIVER_NAMES: tuple[str, ...] = ("libcuda.so.1", "libcuda.so")

_CUDA_LIBRARY_HANDLES: list[ctypes.CDLL] = []

_CUDA_SONAME_FAMILIES: frozenset[str] = frozenset(
    {
        "libcuda",
        "libcudart",
        "libcublas",
        "libcublasLt",
        "libcusparse",
        "libcusolver",
        "libnvJitLink",
        "libnvrtc",
    }
)


@lru_cache(maxsize=1)
def prepare_cuda_libraries() -> tuple[str, ...]:
    """Preload exactly one CUDA userspace stack, preferring system CUDA.

    This keeps the Python package usable in both deployment modes:

    * bare-metal/cloud images with a system CUDA toolkit and no pip
      ``nvidia-*-cu12`` wheels;
    * portable installs where the CUDA userspace stack comes from pip wheels.

    The critical rule is single-stack loading. Loading wheel cuBLAS first on a
    host where system cuBLAS is also reachable can map two distinct files with
    the same SONAME into one process. cuBLAS handle state can then split across
    implementations and later abort in glibc with ``double free or corruption``.
    """

    if sys.platform != "linux":
        return ()
    assert_no_cuda_library_conflicts("preparing CUDA libraries")
    paths = _cuda_library_candidates()
    loaded = _preload_paths(paths)
    assert_no_cuda_library_conflicts("preloading CUDA libraries")
    return loaded


@dataclass(frozen=True)
class CudaDiagnostics:
    """Typed CUDA loader-state snapshot.

    This is the single source of truth for the diagnostics schema. The
    public :func:`cuda_diagnostics` returns ``asdict(...)`` of this so the
    dict surface keeps working, and consumers (formatter, conflict
    warning) normalize whatever they receive through
    :meth:`from_mapping`, which fills defaults for missing fields. That
    keeps tests and downstream callers from breaking when they hand back
    a partial dict, and keeps every key name defined in exactly one
    place.
    """

    platform: str = ""
    mapped: dict[str, list[str]] = field(default_factory=dict)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    packaged_nvidia_roots: list[str] = field(default_factory=list)
    packaged_cuda_library_dirs: list[str] = field(default_factory=list)
    packaged_complete_stacks: list[list[str]] = field(default_factory=list)
    system_driver_libraries: list[str] = field(default_factory=list)
    system_complete_stacks: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, info: Mapping[str, object]) -> "CudaDiagnostics":
        kwargs = {f.name: info[f.name] for f in fields(cls) if f.name in info}
        return cls(**kwargs)


def cuda_diagnostics() -> dict[str, object]:
    """Return CUDA loader state without dlopening any CUDA library."""

    mapped = _mapped_cuda_libraries()
    conflicts = {family: paths for family, paths in mapped.items() if len(paths) > 1}
    return asdict(
        CudaDiagnostics(
            platform=sys.platform,
            mapped=mapped,
            conflicts=conflicts,
            packaged_nvidia_roots=[str(path) for path in _nvidia_roots()],
            packaged_cuda_library_dirs=list(cuda_subprocess_library_dirs()),
            packaged_complete_stacks=[
                [str(path) for path in stack]
                for root in _nvidia_roots()
                if (stack := _complete_nvidia_stack(root))
            ],
            system_driver_libraries=[
                str(path)
                for lib_dir in _system_cuda_driver_dirs()
                if (path := _first_driver_library(lib_dir))
            ],
            system_complete_stacks=[
                [str(path) for path in stack]
                for lib_dir in _system_cuda_lib_dirs()
                if (stack := _complete_system_stack(lib_dir))
            ],
        )
    )


def format_cuda_diagnostics() -> str:
    """Format :func:`cuda_diagnostics` as stable, grep-friendly lines."""

    diag = CudaDiagnostics.from_mapping(cuda_diagnostics())
    lines = ["gamfit CUDA diagnostics:"]
    lines.append(f"  platform: {diag.platform}")
    lines.append("  mapped CUDA libraries:")
    if diag.mapped:
        for family in sorted(diag.mapped):
            lines.append(f"    {family}:")
            for path in diag.mapped[family]:
                lines.append(f"      {path}")
    else:
        lines.append("    <none>")
    lines.append("  CUDA library conflicts:")
    if diag.conflicts:
        for family in sorted(diag.conflicts):
            lines.append(f"    {family}:")
            for path in diag.conflicts[family]:
                lines.append(f"      {path}")
    else:
        lines.append("    <none>")
    lines.append("  packaged nvidia roots:")
    if diag.packaged_nvidia_roots:
        lines.extend(f"    {path}" for path in diag.packaged_nvidia_roots)
    else:
        lines.append("    <none>")
    lines.append("  packaged CUDA library dirs for subprocesses:")
    if diag.packaged_cuda_library_dirs:
        lines.extend(f"    {path}" for path in diag.packaged_cuda_library_dirs)
    else:
        lines.append("    <none>")
    lines.append("  complete packaged CUDA stacks:")
    _append_stack_lines(lines, diag.packaged_complete_stacks)
    lines.append("  system CUDA driver libraries:")
    if diag.system_driver_libraries:
        lines.extend(f"    {path}" for path in diag.system_driver_libraries)
    else:
        lines.append("    <none>")
    lines.append("  complete system CUDA stacks:")
    _append_stack_lines(lines, diag.system_complete_stacks)
    return "\n".join(lines)


def cuda_subprocess_library_dirs() -> tuple[str, ...]:
    """Return packaged CUDA wheel library directories for child processes.

    ``prepare_cuda_libraries`` can discover and preload CUDA userspace
    libraries from pip ``nvidia-*-cu12`` wheels inside the current Python
    process. A subprocess does not inherit those loaded handles; it only sees
    the dynamic loader search path. This helper exposes the corresponding
    wheel ``.../nvidia/<component>/lib`` directories, including
    ``cuda_nvrtc/lib``, so callers can build a subprocess environment without
    guessing uv/pip's unpack location.
    """

    if sys.platform != "linux":
        return ()
    dirs: list[Path] = []
    for root in _nvidia_roots():
        dirs.extend(_nvidia_component_library_dirs(root))
    return tuple(str(path) for path in _dedup_resolved(dirs))


def cuda_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``env`` with packaged CUDA wheel dirs prepended to LD_LIBRARY_PATH."""

    out = dict(os.environ if env is None else env)
    dirs = cuda_subprocess_library_dirs()
    if not dirs:
        return out
    existing = out.get("LD_LIBRARY_PATH", "")
    parts = [*dirs, *(part for part in existing.split(":") if part)]
    out["LD_LIBRARY_PATH"] = ":".join(_dedup_strings(parts))
    return out


_CUDA_CONFLICT_WARNED: set[str] = set()


def assert_no_cuda_library_conflicts(context: str) -> None:
    """Warn (once per process) if multiple CUDA stacks are mapped, then proceed.

    Two distinct files for the same SONAME (e.g. ``libcublas.so.12``) can
    appear in ``/proc/self/maps`` when both a system CUDA install and the
    pip ``nvidia-*-cu12`` wheels are reachable. The catastrophic case —
    cuBLAS handle ownership splitting across implementations and crashing
    on ``cublasDestroy_v2`` — only triggers if calling code does
    ``dlopen(absolute_path)`` on BOTH files and crosses handles. Standard
    library code uses ``dlopen(SONAME)``, which glibc resolves to exactly
    one file via the loader's search path, so all CUDA calls route through
    one handle even though both files are present in the address space.

    This check used to ``raise RuntimeError``. That was overcautious — it
    refused service on common Colab and cloud images where the dual mapping
    is benign. The check now emits one warning per (process, conflict-set)
    and returns. If a real double-free occurs later, the warning makes the
    cause discoverable; if it doesn't (the typical case), gamfit just works.
    """

    diag = CudaDiagnostics.from_mapping(cuda_diagnostics())
    if not diag.conflicts:
        return
    key = "|".join(
        f"{name}:{','.join(sorted(paths))}"
        for name, paths in sorted(diag.conflicts.items())
    )
    if key in _CUDA_CONFLICT_WARNED:
        return
    _CUDA_CONFLICT_WARNED.add(key)
    details = format_cuda_diagnostics()
    sys.stderr.write(
        f"[gamfit] WARNING: dual CUDA stack detected before {context}. "
        "Multiple files share the same SONAME in /proc/self/maps; this is "
        "usually benign (dlopen-by-SONAME deterministically picks one) and "
        "gamfit will proceed. If cublasDestroy_v2 later aborts with "
        "'double free or corruption', keep exactly one CUDA toolkit "
        "reachable: either the system install (typically /usr/local/cuda*) "
        "or the pip nvidia-*-cu12 wheels, not both.\n"
        f"{details}\n"
    )


def _cuda_library_candidates() -> tuple[Path, ...]:
    mapped = _mapped_cuda_libraries()
    if mapped:
        # A CUDA userspace stack may already be live because another package
        # imported cuBLAS/cuSOLVER first. Do not introduce another userspace
        # stack, but still load the driver if it is absent: cudarc needs
        # `libcuda` specifically and preloaded cudart/cuBLAS are not enough.
        if "libcuda" not in mapped and (driver := _cuda_driver_candidate()) is not None:
            return (driver,)
        return ()
    driver = _cuda_driver_candidate()
    if driver is None:
        # CUDA user-space libraries are not useful without the driver. More
        # importantly, preloading cuBLAS/cuSOLVER without libcuda makes the
        # process look partially CUDA-initialized while Rust must still fall
        # back to CPU. Keep the loader state honest and let Rust report a clean
        # CPU fallback.
        return ()
    for lib_dir in _system_cuda_lib_dirs():
        candidates = _complete_system_stack(lib_dir)
        if candidates:
            return tuple(_prepend_driver(driver, candidates))
    for root in _nvidia_roots():
        candidates = _complete_nvidia_stack(root)
        if candidates:
            return tuple(_prepend_driver(driver, candidates))
    return ()


def _preload_paths(paths: Iterable[Path]) -> tuple[str, ...]:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    loaded: list[str] = []
    for path in paths:
        try:
            handle = ctypes.CDLL(str(path), mode=mode)
        except OSError:
            continue
        _CUDA_LIBRARY_HANDLES.append(handle)
        loaded.append(str(path))
    return tuple(loaded)


def _append_stack_lines(lines: list[str], value: object) -> None:
    if not isinstance(value, list) or not value:
        lines.append("    <none>")
        return
    for index, stack in enumerate(value, start=1):
        lines.append(f"    stack {index}:")
        if isinstance(stack, list):
            lines.extend(f"      {path}" for path in stack)


def _mapped_cuda_libraries() -> dict[str, list[str]]:
    if sys.platform != "linux":
        return {}
    maps = Path("/proc/self/maps")
    try:
        content = maps.read_text()
    except OSError:
        return {}
    grouped: dict[str, set[str]] = {}
    for line in content.splitlines():
        fields = line.split()
        if not fields:
            continue
        raw_path = fields[-1]
        if not raw_path.startswith("/"):
            continue
        family = _cuda_library_family(Path(raw_path).name)
        if family is None:
            continue
        grouped.setdefault(family, set()).add(raw_path)
    return {family: sorted(paths) for family, paths in sorted(grouped.items())}


def _cuda_library_family(basename: str) -> str | None:
    stem = basename.split(".so", 1)[0]
    return stem if stem in _CUDA_SONAME_FAMILIES else None


def _nvidia_roots() -> tuple[Path, ...]:
    if sys.platform != "linux":
        return ()
    roots: list[Path] = []
    spec = util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations is not None:
        roots.extend(Path(location) for location in spec.submodule_search_locations)
    return tuple(_existing_unique_paths(roots))


def _system_cuda_lib_dirs() -> tuple[Path, ...]:
    if sys.platform != "linux":
        return ()
    roots: list[Path] = [
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/lib"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib/wsl/lib"),
    ]
    try:
        versioned = sorted(
            path
            for path in Path("/usr/local").iterdir()
            if path.name.startswith("cuda-") and path.is_dir()
        )
    except OSError:
        versioned = []
    for root in versioned:
        roots.append(root / "lib64")
        roots.append(root / "lib")
        roots.append(root / "targets" / "x86_64-linux" / "lib")
    return tuple(_existing_unique_paths(roots))


def _system_cuda_driver_dirs() -> tuple[Path, ...]:
    if sys.platform != "linux":
        return ()
    roots: list[Path] = [
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/local/nvidia/lib"),
        Path("/usr/local/cuda/compat"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib/wsl/lib"),
    ]
    return tuple(_existing_unique_paths(roots))


def _cuda_driver_candidate() -> Path | None:
    for lib_dir in _system_cuda_driver_dirs():
        path = _first_driver_library(lib_dir)
        if path is not None:
            return path
    return None


def _first_driver_library(lib_dir: Path) -> Path | None:
    direct = _first_existing(lib_dir, _CUDA_DRIVER_NAMES)
    if direct is not None:
        return direct
    try:
        matches = sorted(lib_dir.glob("libcuda.so.*"))
    except OSError:
        return None
    for path in matches:
        if path.name != "libcuda.so" and path.exists():
            return path
    return None


def _prepend_driver(driver: Path | None, stack: list[Path]) -> list[Path]:
    if driver is None:
        return stack
    return _dedup_resolved([driver, *stack])


def _complete_nvidia_stack(root: Path) -> list[Path] | None:
    out: list[Path] = []
    for component, library_groups in _CUDA_LIBRARY_GROUPS:
        lib_dir = root / component / "lib"
        for names in library_groups:
            path = _first_existing(lib_dir, names)
            if path is None:
                return None
            out.append(path)
    return _dedup_resolved(out)


def _nvidia_component_library_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for component, library_groups in _CUDA_SUBPROCESS_LIBRARY_GROUPS:
        lib_dir = root / component / "lib"
        if any(_first_existing(lib_dir, names) is not None for names in library_groups):
            out.append(lib_dir)
    return _existing_unique_paths(out)


def _complete_system_stack(lib_dir: Path) -> list[Path] | None:
    out: list[Path] = []
    for _, library_groups in _CUDA_LIBRARY_GROUPS:
        for names in library_groups:
            path = _first_existing(lib_dir, names)
            if path is None:
                return None
            out.append(path)
    return _dedup_resolved(out)


def _first_existing(lib_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = lib_dir / name
        if path.exists():
            return path
    return None


def _dedup_resolved(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _existing_unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _dedup_strings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out

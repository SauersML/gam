from __future__ import annotations

import sys
import ctypes
import os
from functools import lru_cache
from importlib import util
from pathlib import Path
from typing import Iterable

_CUDA_LIBRARY_GROUPS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("cuda_runtime", (("libcudart.so.12", "libcudart.so"),)),
    ("nvjitlink", (("libnvJitLink.so.12", "libnvJitLink.so"),)),
    (
        "cublas",
        (
            ("libcublasLt.so.12", "libcublasLt.so"),
            ("libcublas.so.12", "libcublas.so"),
        ),
    ),
    ("cusparse", (("libcusparse.so.12", "libcusparse.so"),)),
    ("cusolver", (("libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"),)),
)

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


def cuda_diagnostics() -> dict[str, object]:
    """Return CUDA loader state without dlopening any CUDA library."""

    mapped = _mapped_cuda_libraries()
    conflicts = {family: paths for family, paths in mapped.items() if len(paths) > 1}
    return {
        "platform": sys.platform,
        "mapped": mapped,
        "conflicts": conflicts,
        "packaged_nvidia_roots": [str(path) for path in _nvidia_roots()],
        "packaged_complete_stacks": [
            [str(path) for path in stack]
            for root in _nvidia_roots()
            if (stack := _complete_nvidia_stack(root))
        ],
        "system_complete_stacks": [
            [str(path) for path in stack]
            for lib_dir in _system_cuda_lib_dirs()
            if (stack := _complete_system_stack(lib_dir))
        ],
    }


def format_cuda_diagnostics() -> str:
    """Format :func:`cuda_diagnostics` as stable, grep-friendly lines."""

    info = cuda_diagnostics()
    lines = ["gamfit CUDA diagnostics:"]
    lines.append(f"  platform: {info['platform']}")
    lines.append("  mapped CUDA libraries:")
    mapped = info["mapped"]
    if isinstance(mapped, dict) and mapped:
        for family in sorted(mapped):
            lines.append(f"    {family}:")
            for path in mapped[family]:
                lines.append(f"      {path}")
    else:
        lines.append("    <none>")
    lines.append("  CUDA library conflicts:")
    conflicts = info["conflicts"]
    if isinstance(conflicts, dict) and conflicts:
        for family in sorted(conflicts):
            lines.append(f"    {family}:")
            for path in conflicts[family]:
                lines.append(f"      {path}")
    else:
        lines.append("    <none>")
    lines.append("  packaged nvidia roots:")
    packaged_roots = info["packaged_nvidia_roots"]
    if isinstance(packaged_roots, list) and packaged_roots:
        lines.extend(f"    {path}" for path in packaged_roots)
    else:
        lines.append("    <none>")
    lines.append("  complete packaged CUDA stacks:")
    _append_stack_lines(lines, info["packaged_complete_stacks"])
    lines.append("  complete system CUDA stacks:")
    _append_stack_lines(lines, info["system_complete_stacks"])
    return "\n".join(lines)


def assert_no_cuda_library_conflicts(context: str) -> None:
    """Raise before Rust/CUDA work starts if multiple CUDA stacks are mapped."""

    conflicts = cuda_diagnostics()["conflicts"]
    if not isinstance(conflicts, dict) or not conflicts:
        return
    details = format_cuda_diagnostics()
    raise RuntimeError(
        f"CUDA library conflict before {context}. Multiple distinct shared objects "
        "for the same CUDA SONAME family are already mapped in this Python process. "
        "glibc deduplicates dlopen by file identity rather than SONAME, so cuBLAS "
        "handle ownership can split across implementations and abort with "
        f"'double free or corruption (!prev)'.\n{details}"
    )


def _cuda_library_candidates() -> tuple[Path, ...]:
    mapped = _mapped_cuda_libraries()
    if mapped:
        # A CUDA stack is already live. Do not introduce another one.
        return ()
    for lib_dir in _system_cuda_lib_dirs():
        candidates = _complete_system_stack(lib_dir)
        if candidates:
            return tuple(candidates)
    for root in _nvidia_roots():
        candidates = _complete_nvidia_stack(root)
        if candidates:
            return tuple(candidates)
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

from __future__ import annotations

import ctypes
import os
import sys
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
    ("cusolver", (("libcusolver.so.11", "libcusolver.so"),)),
)

_CUDA_LIBRARY_HANDLES: list[ctypes.CDLL] = []


@lru_cache(maxsize=1)
def preload_cuda_libraries() -> tuple[str, ...]:
    """Preload CUDA wheel libraries before Rust's runtime probe runs."""

    if sys.platform != "linux":
        return ()

    mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    loaded: list[str] = []
    for path in _cuda_library_candidates():
        try:
            handle = ctypes.CDLL(str(path), mode=mode)
        except OSError:
            continue
        _CUDA_LIBRARY_HANDLES.append(handle)
        loaded.append(str(path))
    return tuple(loaded)


def _cuda_library_candidates() -> tuple[Path, ...]:
    for root in _nvidia_roots():
        candidates = _complete_nvidia_stack(root)
        if candidates:
            return candidates
    for lib_dir in _system_cuda_lib_dirs():
        candidates = _complete_system_stack(lib_dir)
        if candidates:
            return candidates
    return ()


def _complete_nvidia_stack(root: Path) -> tuple[Path, ...]:
    out: list[Path] = []
    for component, library_groups in _CUDA_LIBRARY_GROUPS:
        lib_dir = root / component / "lib"
        for names in library_groups:
            path = _first_existing(lib_dir, names)
            if path is None:
                return ()
            out.append(path)
    return tuple(_dedup_resolved(out))


def _complete_system_stack(lib_dir: Path) -> tuple[Path, ...]:
    out: list[Path] = []
    for _, library_groups in _CUDA_LIBRARY_GROUPS:
        for names in library_groups:
            path = _first_existing(lib_dir, names)
            if path is None:
                return ()
            out.append(path)
    return tuple(_dedup_resolved(out))


def _first_existing(lib_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = lib_dir / name
        if path.exists():
            return path
    return None


def _dedup_resolved(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return tuple(out)


def _nvidia_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    spec = util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations is not None:
        roots.extend(Path(location) for location in spec.submodule_search_locations)
    return tuple(_existing_unique_paths(roots))


def _system_cuda_lib_dirs() -> tuple[Path, ...]:
    return _existing_unique_paths((Path("/usr/local/cuda/lib64"), Path("/usr/local/cuda/lib")))


def _existing_unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return tuple(out)

"""Locator for the compiled `gamfit._rust` extension shared by bench scripts.

The bench scripts deliberately avoid `import gamfit._rust` because the
package facade in `gamfit/__init__.py` does heavy work at import time
(CUDA discovery, NumPy/Pandas validation, etc.). Instead they glob for
the compiled `_rust*.so` (or `_rust*.pyd` on Windows) and load it with
`importlib.util.spec_from_file_location`.

The naive implementation — look up `find_spec("gamfit").submodule_search_locations`
and glob those — has a subtle failure mode when:

* the user has both a source checkout AND a pip-installed wheel of `gamfit`, and
* sys.path is arranged so the source tree shadows the wheel (this is exactly
  what `bench/fuzz_vs_mgcv.py` does via `sys.path.insert(0, str(ROOT))`).

In that configuration the importer resolves `gamfit` to the source tree (a
regular package with `__init__.py` and no compiled extension), so
`submodule_search_locations` is a single-element list pointing at the source
tree, which has no `_rust*.so`. The wheel-installed `.so` sitting in
site-packages is invisible to the loader and the fuzzer aborts with
"gamfit Rust extension is not built", even though `pip install` of the maturin
wheel succeeded earlier in the same CI step.

This loader fixes that by enumerating every plausible location:

1. The importable `gamfit` package's own `submodule_search_locations` (covers
   the common `maturin develop` case where the source tree owns the `.so`).
2. Every `sys.path` entry's `gamfit/` subdirectory (covers pip-installed wheels
   in site-packages even when the source tree shadows them on the import path).
3. The installed distribution's files via `importlib.metadata` (covers wheels
   installed to non-default site dirs).
4. The repo-relative `ROOT/gamfit/` source tree (final fallback).

Whichever location actually contains `_rust*.so` / `_rust*.pyd` wins.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import typing
from pathlib import Path

_CACHED_MODULE: typing.Any = None


def _scan_dir(d: Path, candidates: list[Path], seen: set[Path]) -> None:
    if d in seen:
        return
    seen.add(d)
    if not d.is_dir():
        return
    candidates.extend(sorted(d.glob("_rust*.so")))
    candidates.extend(sorted(d.glob("_rust*.pyd")))


def find_rust_extension(source_tree_root: Path | None = None) -> Path | None:
    """Return the path to the compiled `gamfit._rust` extension, or None.

    `source_tree_root` is the repo root (the directory containing the
    in-tree `gamfit/`); callers pass `ROOT` so this helper stays
    location-agnostic.
    """

    candidates: list[Path] = []
    seen: set[Path] = set()

    pkg_spec = importlib.util.find_spec("gamfit")
    if pkg_spec is not None and pkg_spec.submodule_search_locations:
        for d in pkg_spec.submodule_search_locations:
            _scan_dir(Path(d), candidates, seen)

    for entry in sys.path:
        if not entry:
            continue
        _scan_dir(Path(entry) / "gamfit", candidates, seen)

    try:
        dist = importlib.metadata.distribution("gamfit")
    except importlib.metadata.PackageNotFoundError:
        dist = None
    if dist is not None and dist.files is not None:
        for f in dist.files:
            name = Path(f.name).name
            if not (name.startswith("_rust") and (name.endswith(".so") or name.endswith(".pyd"))):
                continue
            located = Path(dist.locate_file(f))
            if located.is_file() and located not in candidates:
                candidates.append(located)

    if source_tree_root is not None:
        _scan_dir(source_tree_root / "gamfit", candidates, seen)

    return candidates[0] if candidates else None


def load_gamfit_rust_module(source_tree_root: Path | None = None) -> typing.Any:
    """Resolve and load the compiled `gamfit._rust` extension."""

    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE
    located = find_rust_extension(source_tree_root)
    if located is None:
        raise RuntimeError("gamfit Rust extension is not built")
    spec = importlib.util.spec_from_file_location("_rust", located)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load gamfit Rust extension from {located}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CACHED_MODULE = module
    return module

#!/usr/bin/env python3
"""Keep every gamfit release small enough that PyPI can go on holding all of them.

PyPI keeps every file of every release, yanked ones included, and refuses an
upload once the project's total would pass its size limit. twine uploads one
file at a time, so hitting the limit mid-upload publishes a release that has
some wheels and not others. This script guards both ends (RELEASING.md):

``sdist SDIST``
    Unpack the source distribution, compile it exactly as shipped with
    ``cargo check --locked --workspace``, and require that every file it
    carries is either a source the compiler read (rustc's dep-info, which
    lists each module, ``include_str!`` and ``include_bytes!`` input) or
    packaging metadata (manifests, the lockfile, readme and license, the
    Python package). A test module, fixture, example or benchmark is not read
    by a library build, so it is refused, and a source that an exclude glob in
    pyproject.toml wrongly dropped fails the compile. ``--locked`` also proves
    the shipped Cargo.lock still describes the shipped workspace.

    Modules compiled only for another target (``#[cfg(windows)]``) would read
    as unused here; the tree has none.

``storage DIST_DIR``
    Before any upload: ask the index what the project already holds, add every
    file in DIST_DIR it does not have yet (the upload runs with
    ``--skip-existing``), and refuse the release if that would take the project
    past ``HEADROOM`` of its size limit, or if any single file is over PyPI's
    per-file limit.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request

GIB = 2**30
MIB = 2**20

# PyPI's defaults (warehouse `MAX_PROJECT_SIZE` and `MAX_FILESIZE`). An
# increase granted on request is not visible through any API, so the project
# limit is an input: the release workflow passes the repository variable
# PYPI_PROJECT_SIZE_LIMIT_GIB when one has been granted.
PYPI_DEFAULT_PROJECT_LIMIT = 10 * GIB
PYPI_FILE_LIMIT = 100 * MIB

# A release is refused once it would leave less than a tenth of the limit
# free. At the current release size (seven wheels and the sdist, ~250 MB; the
# free-threaded wheels add two more per platform) that is room for at least one
# more release, so the refusal arrives while pruning old pre-releases or asking
# PyPI for more space is still a routine chore rather than a half-published
# release.
HEADROOM = 0.9

SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"
PYPI_SIMPLE = "https://pypi.org/simple/"


# ------------------------------------------------------------------------ sdist
def parse_dep_info(text: str) -> set[str]:
    """Every prerequisite named in a Makefile-style dep-info file from rustc."""
    paths: set[str] = set()
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        # `target: dep dep ...`; spaces inside a path are escaped as `\ `.
        _, sep, deps = line.partition(": ")
        if not sep:
            continue
        paths.update(part.replace("\\ ", " ") for part in re.split(r"(?<!\\) ", deps) if part)
    return paths


def read_by_compiler(target_dir: pathlib.Path, tree: pathlib.Path) -> set[str]:
    """Tree-relative paths of the files rustc read, from every dep-info file."""
    tree = tree.resolve()
    read: set[str] = set()
    for dep_info in target_dir.rglob("*.d"):
        for path in parse_dep_info(dep_info.read_text()):
            # Workspace members are compiled with paths relative to the
            # workspace root, and `#[path = "../x.rs"]` leaves `..` in them.
            full = pathlib.Path(os.path.normpath(tree / path))
            if full.is_relative_to(tree):
                read.add(full.relative_to(tree).as_posix())
    return read


def is_packaging_metadata(path: str, python_package: str) -> bool:
    """Files the sdist needs for what they declare, not for what rustc reads."""
    parts = pathlib.PurePosixPath(path).parts
    if parts[-1] == "Cargo.toml":
        return True
    if len(parts) == 1:
        return parts[0] in {"PKG-INFO", "pyproject.toml", "Cargo.lock", "rust-toolchain.toml"} or parts[
            0
        ].startswith(("README", "LICENSE"))
    # The Python package: its contents are checked where they ship, in the
    # wheel (scripts/wheel_smoke.py).
    return parts[0] == python_package


def unread_files(members: list[str], read: set[str], python_package: str) -> list[str]:
    return sorted(m for m in members if m not in read and not is_packaging_metadata(m, python_package))


def sdist_members(sdist: pathlib.Path) -> tuple[str, list[str]]:
    """The sdist's top-level directory and its files, relative to that directory."""
    with tarfile.open(sdist) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {pathlib.PurePosixPath(name).parts[0] for name in names}
    if len(roots) != 1:
        raise SystemExit(f"{sdist} must hold one top-level directory, found {sorted(roots)}")
    (root,) = roots
    return root, sorted(pathlib.PurePosixPath(name).relative_to(root).as_posix() for name in names)


def check_sdist(sdist: pathlib.Path, target_dir: pathlib.Path) -> None:
    root, members = sdist_members(sdist)
    with tempfile.TemporaryDirectory(prefix="gamfit-sdist-") as scratch:
        with tarfile.open(sdist) as archive:
            archive.extractall(scratch, filter="data")
        tree = pathlib.Path(scratch) / root
        with (tree / "pyproject.toml").open("rb") as handle:
            module = tomllib.load(handle)["tool"]["maturin"]["module-name"]
        subprocess.run(
            ["cargo", "check", "--locked", "--workspace", "--target-dir", str(target_dir)],
            cwd=tree,
            check=True,
        )
        unread = unread_files(members, read_by_compiler(target_dir, tree), module.split(".")[0])
    size = sdist.stat().st_size
    print(f"{sdist.name}: {size / MIB:.2f} MiB, {len(members)} files, compiles with its Cargo.lock")
    if unread:
        raise SystemExit(
            f"{sdist.name} ships {len(unread)} files the build never reads; exclude them under "
            "[tool.maturin] in pyproject.toml:\n  " + "\n  ".join(unread)
        )


# ---------------------------------------------------------------------- storage
def project_name(dist_dir: pathlib.Path) -> tuple[str, list[pathlib.Path]]:
    files = sorted(path for path in dist_dir.iterdir() if path.is_file())
    if not files:
        raise SystemExit(f"{dist_dir} holds no distributions")
    names = {re.sub(r"[-_.]+", "-", path.name.split("-")[0]).lower() for path in files}
    if len(names) != 1:
        raise SystemExit(f"{dist_dir} mixes distributions of {sorted(names)}")
    return names.pop(), files


def published_files(index_url: str, project: str) -> dict[str, int]:
    """Filename -> size of every file the index holds for ``project`` (PEP 691/700)."""
    request = urllib.request.Request(f"{index_url.rstrip('/')}/{project}/", headers={"Accept": SIMPLE_JSON})
    with urllib.request.urlopen(request) as response:
        page = json.load(response)
    return {entry["filename"]: entry["size"] for entry in page["files"]}


def storage_verdict(
    published: dict[str, int], new: dict[str, int], limit: int
) -> tuple[list[str], str]:
    """The reasons to refuse this upload (empty when it may proceed), and a usage line."""
    problems = [
        f"{name} is {size / MIB:.1f} MiB; PyPI refuses files over {PYPI_FILE_LIMIT / MIB:.0f} MiB"
        for name, size in sorted(new.items())
        if size > PYPI_FILE_LIMIT
    ]
    used = sum(published.values())
    adding = sum(size for name, size in new.items() if name not in published)
    after = used + adding
    usage = (
        f"PyPI holds {used / GIB:.2f} GiB in {len(published)} files; this release adds "
        f"{adding / GIB:.3f} GiB, reaching {after / GIB:.2f} GiB of {limit / GIB:.2f} GiB ({after / limit:.1%})"
    )
    if after > HEADROOM * limit:
        problems.append(
            f"{usage}, past the {HEADROOM:.0%} release threshold; prune old pre-releases or "
            "raise the limit first (RELEASING.md)"
        )
    return problems, usage


def check_storage(dist_dir: pathlib.Path, index_url: str, limit_gib: float) -> None:
    project, files = project_name(dist_dir)
    published = published_files(index_url, project)
    problems, usage = storage_verdict(published, {path.name: path.stat().st_size for path in files}, round(limit_gib * GIB))
    if problems:
        raise SystemExit("\n".join(problems))
    print(usage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sdist = sub.add_parser("sdist", help="compile the sdist as shipped and refuse files the build never reads")
    sdist.add_argument("sdist", type=pathlib.Path)
    sdist.add_argument("--target-dir", type=pathlib.Path, required=True)
    storage = sub.add_parser("storage", help="refuse an upload that would fill the PyPI project")
    storage.add_argument("dist_dir", type=pathlib.Path)
    storage.add_argument("--index-url", default=PYPI_SIMPLE)
    storage.add_argument("--project-limit-gib", type=float, default=PYPI_DEFAULT_PROJECT_LIMIT / GIB)
    args = parser.parse_args(argv)

    if args.command == "sdist":
        check_sdist(args.sdist, args.target_dir.resolve())
    else:
        check_storage(args.dist_dir, args.index_url, args.project_limit_gib)


if __name__ == "__main__":
    sys.exit(main())

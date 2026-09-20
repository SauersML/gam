#!/usr/bin/env python3
"""The gamfit version of a build, derived from the gam commit it is built from (gam#3157).

The release line is ``pyproject.toml``'s ``version = "R"``, the line build.rs holds gam-pyffi's Cargo.toml
and uv.lock to. The commit that set it is that release's engine. Every other tree is a different engine,
so it must not carry the same version string. Before this, every commit of main after 9c0e6b4845 built
under one version string, across a removed public function (``set_log_level``) and saved-model payloads
16 to 29, and a caller pinned by version could not tell them apart. The version of a build is:

``R``
    the tree is clean and its commit set ``version = "R"``, meaning every parent carried another version.
    This is the release engine.
``MAJOR.MINOR.(PATCH+1).devD+gCOMMIT``, with ``.dirty`` appended when tracked files differ from ``COMMIT``
    any other tree. ``COMMIT`` is the full hash that ``gamfit.build_info()["commit"]`` records. ``D`` counts
    the commits reachable from ``COMMIT`` and not from the commit(s) that set ``R``. Every release bump of
    ``R`` is at least its next patch release, so the build sorts after ``R`` and before the release that
    follows it. Its public part ``...devD`` differs from every release, so a ``gamfit==R`` pin never
    matches it.

maturin reads the version statically and has no override (maturin#1283, maturin#2163). Nothing here
edits a tracked file, which would also flip the dirty state the engine records. Instead, the derived
version is written into what the build produced:

``derive``
    print the version of the tree at ``--root``.
``release``
    print the release version the tree is, or refuse when it derives a development version. A
    publication gates on this, so only the engine that set the release line can be published under it.
``stamp WHEEL``
    rewrite a built gamfit wheel to that version: the dist-info name, METADATA, RECORD and the file name.
``stamp-installed``
    rewrite the gamfit that ``maturin develop`` installed into the running interpreter, after checking
    that the installed engine recorded the tree's commit and dirty state.

A shallow clone cannot see the commit that set ``R``, and a tree outside a gam checkout has no commit.
Both are refused rather than given a guessed version.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import pathlib
import re
import subprocess
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = "pyproject.toml"
DISTRIBUTION = "gamfit"

_RELEASE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_VERSION_LINE = re.compile(r'^version = \s*"([^"]*)"')


class VersionError(RuntimeError):
    """The build's version cannot be derived or written, with the reason."""


# --------------------------------------------------------------------- derive
def _git(root: pathlib.Path, *args: str) -> str:
    """Run git in ``root`` without taking the optional index lock, as gam-build-identity does."""
    done = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True)
    if done.returncode != 0:
        stderr = done.stderr.decode("utf-8", "replace").strip()
        raise VersionError(f"`git {' '.join(args)}` failed in {root}: {stderr}")
    return done.stdout.decode("utf-8")


def declared_version(text: bytes) -> str | None:
    """The release line a pyproject.toml declares, read the way build.rs reads it; None when there is none.

    build.rs's ``read_toml_version_line`` takes the first line that starts with ``version = "`` and holds
    gam-pyffi's Cargo.toml and uv.lock to it, so this is the one definition of the release line.
    """
    for line in text.decode("utf-8", "replace").splitlines():
        match = _VERSION_LINE.match(line.strip())
        if match:
            return match.group(1)
    return None


def _version_at(root: pathlib.Path, commit: str) -> str | None:
    """The version pyproject.toml declares at ``commit``; None when the file is absent there."""
    done = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "show", f"{commit}:{PYPROJECT}"],
        capture_output=True,
    )
    return declared_version(done.stdout) if done.returncode == 0 else None


def release_commits(root: pathlib.Path, release: str) -> list[str]:
    """The commits in HEAD's history that set ``version = release``.

    A commit sets the version when its pyproject.toml declares it and no parent's does. The walk runs over
    pyproject.toml's simplified history (``git rev-list --parents -- pyproject.toml``), whose rewritten
    parent of a commit is the nearest ancestor that changed the file. Everything in between is identical
    to that ancestor, so comparing against rewritten parents is comparing against real ones. A merge that
    kept one side's file follows only that side, the side HEAD's file came from.
    """
    history = _git(root, "rev-list", "--parents", "--topo-order", "HEAD", "--", PYPROJECT)
    lines = [line.split() for line in history.splitlines()]
    if not lines:
        raise VersionError(f"no commit in HEAD's history of {root} adds {PYPROJECT}")
    parents = {line[0]: line[1:] for line in lines}
    versions: dict[str, str | None] = {}

    def version(commit: str) -> str | None:
        if commit not in versions:
            versions[commit] = _version_at(root, commit)
        return versions[commit]

    # --topo-order lists no commit before its children, so the first is the one HEAD's file came from.
    newest = lines[0][0]
    if version(newest) != release:
        raise VersionError(
            f"{newest}, the last commit to change {PYPROJECT} up to HEAD, declares {version(newest)!r}, "
            f"not HEAD's {release!r}"
        )
    setters: list[str] = []
    seen: set[str] = set()
    frontier = [newest]
    while frontier:
        commit = frontier.pop()
        if commit in seen:
            continue
        seen.add(commit)
        carried = [parent for parent in parents[commit] if version(parent) == release]
        if carried:
            frontier.extend(carried)
        else:
            setters.append(commit)
    return setters


def derive(root: pathlib.Path) -> tuple[str, str, bool]:
    """The gamfit version of the tree at ``root`` (see the module docstring), its commit and dirty state."""
    try:
        _git(root, "ls-files", "--error-unmatch", PYPROJECT)
    except VersionError as error:
        raise VersionError(
            f"{root} is not a gam git checkout that tracks {PYPROJECT}, so it has no commit to derive "
            "the gamfit version from"
        ) from error
    if _git(root, "rev-parse", "--is-shallow-repository").strip() == "true":
        raise VersionError(
            f"{root} is a shallow clone, which can cut off the commit that set the release line; fetch "
            "the full history (actions/checkout: fetch-depth: 0, with filter: blob:none to skip old blobs)"
        )
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no").strip())
    release = _version_at(root, commit)
    if release is None:
        raise VersionError(f"{PYPROJECT} at {commit} declares no release line")
    parts = _RELEASE.fullmatch(release)
    if parts is None:
        raise VersionError(
            f"{PYPROJECT} at {commit} declares version {release!r}, not a MAJOR.MINOR.PATCH release line"
        )
    setters = release_commits(root, release)
    if setters == [commit] and not dirty:
        return release, commit, dirty
    distance = int(_git(root, "rev-list", "--count", commit, "--not", *setters))
    major, minor, patch = (int(part) for part in parts.groups())
    version = f"{major}.{minor}.{patch + 1}.dev{distance}+g{commit}" + (".dirty" if dirty else "")
    return version, commit, dirty


def release(root: pathlib.Path) -> str:
    """The release version the tree at ``root`` is; refused for every other tree."""
    version, commit, dirty = derive(root)
    if not _RELEASE.fullmatch(version):
        raise VersionError(
            f"{commit}{' with uncommitted changes' if dirty else ''} is not the commit that set the release "
            f"line, so it builds as {version}; publish the commit that sets the new version instead"
        )
    return version


# ---------------------------------------------------------------------- stamp
def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}"


def _restamp_metadata(text: str, old: str, new: str) -> str:
    header, blank, body = text.partition("\n\n")
    lines = header.split("\n")
    names = [line for line in lines if line.startswith("Name: ")]
    versions = [index for index, line in enumerate(lines) if line.startswith("Version: ")]
    if names != [f"Name: {DISTRIBUTION}"] or len(versions) != 1 or lines[versions[0]] != f"Version: {old}":
        raise VersionError(f"METADATA does not name exactly {DISTRIBUTION} {old}")
    lines[versions[0]] = f"Version: {new}"
    return "\n".join(lines) + blank + body


def _record(entries: list[tuple[str, bytes]], record_path: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for path, data in entries:
        writer.writerow([path, _record_hash(data), len(data)])
    writer.writerow([record_path, "", ""])
    return buffer.getvalue().encode("utf-8")


def stamp_wheel(wheel: pathlib.Path, version: str) -> pathlib.Path:
    """Rewrite a gamfit wheel to ``version`` and return its new path, which replaces ``wheel``."""
    fields = wheel.name[: -len(".whl")].split("-") if wheel.name.endswith(".whl") else []
    if len(fields) not in (5, 6) or fields[0] != DISTRIBUTION:
        raise VersionError(f"{wheel} is not a {DISTRIBUTION} wheel file name")
    old = fields[1]
    if old == version:
        return wheel
    old_prefixes = (f"{DISTRIBUTION}-{old}.dist-info/", f"{DISTRIBUTION}-{old}.data/")
    new_prefixes = (f"{DISTRIBUTION}-{version}.dist-info/", f"{DISTRIBUTION}-{version}.data/")
    record_path = f"{new_prefixes[0]}RECORD"
    with zipfile.ZipFile(wheel) as source:
        infos = source.infolist()
        if f"{old_prefixes[0]}METADATA" not in source.namelist():
            raise VersionError(f"{wheel} has no {old_prefixes[0]}METADATA")
        members: list[tuple[zipfile.ZipInfo, bytes]] = []
        for info in infos:
            name = info.filename
            if name == f"{old_prefixes[0]}RECORD":
                continue
            data = source.read(info)
            for old_prefix, new_prefix in zip(old_prefixes, new_prefixes):
                if name.startswith(old_prefix):
                    name = new_prefix + name[len(old_prefix):]
            if name == f"{new_prefixes[0]}METADATA":
                data = _restamp_metadata(data.decode("utf-8"), old, version).encode("utf-8")
            renamed = zipfile.ZipInfo(name, date_time=info.date_time)
            renamed.external_attr = info.external_attr
            renamed.compress_type = info.compress_type
            renamed.create_system = info.create_system
            members.append((renamed, data))
    record = _record([(info.filename, data) for info, data in members if not info.is_dir()], record_path)
    target = wheel.with_name("-".join([DISTRIBUTION, version, *fields[2:]]) + ".whl")
    handle, scratch = tempfile.mkstemp(dir=wheel.parent, prefix=".stamp-", suffix=".whl")
    os.close(handle)
    try:
        with zipfile.ZipFile(scratch, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for info, data in members:
                out.writestr(info, data)
            record_info = zipfile.ZipInfo(record_path, date_time=members[-1][0].date_time)
            record_info.external_attr = 0o644 << 16
            record_info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(record_info, record)
        os.replace(scratch, target)
    except BaseException:
        os.unlink(scratch)
        raise
    wheel.unlink()
    return target


def stamp_installed(root: pathlib.Path) -> pathlib.Path:
    """Rewrite the gamfit installed in this interpreter to the version of the tree at ``root``.

    The installed engine must record the tree's commit and dirty state, so the version names the engine
    that was built rather than a tree that moved on while it compiled. Returns the new dist-info path.
    """
    from importlib import metadata

    import gamfit

    version, commit, dirty = derive(root)
    info = gamfit.build_info()
    if not info.get("available"):
        raise VersionError(f"the installed gamfit extension did not load: {info.get('reason')}")
    if (info["commit"], info["dirty"]) != (commit, dirty):
        raise VersionError(
            f"the installed engine records commit {info['commit']} dirty={info['dirty']}, but {root} is at "
            f"{commit} dirty={dirty}: the tree changed during the build, so rebuild"
        )
    dist = metadata.distribution(DISTRIBUTION)
    old = dist.version
    listed = [path for path in dist.files or [] if len(path.parts) == 2 and path.parts[1] == "METADATA"]
    if len(listed) != 1 or not listed[0].parts[0].endswith(".dist-info"):
        raise VersionError(f"the installed {DISTRIBUTION} has no single dist-info METADATA in its RECORD")
    old_dir = pathlib.Path(dist.locate_file(listed[0].parts[0]))
    if old == version:
        return old_dir
    site = old_dir.parent
    new_dir = site / f"{DISTRIBUTION}-{version}.dist-info"
    restamped = _restamp_metadata((old_dir / "METADATA").read_text(encoding="utf-8"), old, version)
    with (old_dir / "RECORD").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    os.rename(old_dir, new_dir)
    (new_dir / "METADATA").write_text(restamped, encoding="utf-8")
    old_prefix, new_prefix = f"{old_dir.name}/", f"{new_dir.name}/"
    record_path = f"{new_prefix}RECORD"
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        if not row or not row[0].startswith(old_prefix):
            writer.writerow(row)
            continue
        path = new_prefix + row[0][len(old_prefix):]
        if path == record_path:
            continue
        data = (site / path).read_bytes()
        writer.writerow([path, _record_hash(data), len(data)])
    writer.writerow([record_path, "", ""])
    (new_dir / "RECORD").write_text(buffer.getvalue(), encoding="utf-8")
    return new_dir


# ------------------------------------------------------------------------ cli
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    derive_parser = sub.add_parser("derive", help="print the gamfit version of a gam tree")
    derive_parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    release_parser = sub.add_parser("release", help="print the release the tree is, or refuse")
    release_parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    stamp_parser = sub.add_parser("stamp", help="rewrite built gamfit wheels to the tree's version")
    stamp_parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    stamp_parser.add_argument("wheels", type=pathlib.Path, nargs="+")
    installed_parser = sub.add_parser(
        "stamp-installed", help="rewrite the gamfit installed in this interpreter by maturin develop"
    )
    installed_parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "derive":
            print(derive(args.root)[0])
        elif args.command == "release":
            print(release(args.root))
        elif args.command == "stamp":
            version = derive(args.root)[0]
            for wheel in args.wheels:
                print(stamp_wheel(wheel, version))
        else:
            print(stamp_installed(args.root))
    except VersionError as error:
        raise SystemExit(f"gamfit_version: {error}") from error


if __name__ == "__main__":
    main()

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

An unpacked sdist has no git history, so the version is derived when the sdist is built and recorded
in its PKG-INFO. A build from the sdist reads it back, after checking that it is ``R`` or a development
version of ``R``. A shallow clone cannot see the commit that set ``R``, and a tree that is neither a gam
checkout nor a gamfit sdist records no version. Both are refused rather than given a guessed version.

maturin gives the unstamped version (gam-pyffi's Cargo.toml version, which build.rs holds to ``R``) and
has no override (maturin#1283, maturin#2163). Nothing here edits a tracked file, which would also flip
the dirty state the engine records. Instead, the derived version is written into what the build
produced:

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

The module is also the project's PEP 517 build backend (pyproject.toml ``[build-system]``, with
``backend-path = ["scripts"]``), so ``pip install .``, ``pip install git+…``, ``python -m build`` and
editable installs are stamped too. Every hook delegates to maturin's, then writes the derived version
into the wheel, the sdist (its name, top directory and PKG-INFO) or the prepared dist-info. A build
whose tree changes between the derivation and the end of maturin's work is refused.
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
import tarfile
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


def _next_development(release: str) -> str:
    major, minor, patch = (int(part) for part in _RELEASE.fullmatch(release).groups())
    return f"{major}.{minor}.{patch + 1}"


def _metadata_fields(text: str) -> tuple[list[str], dict[str, list[int]]]:
    """A METADATA or PKG-INFO header's lines and, per field name, the indexes of its lines."""
    lines = text.partition("\n\n")[0].split("\n")
    fields: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        name, colon, _ = line.partition(": ")
        if colon and not line.startswith((" ", "\t")):
            fields.setdefault(name, []).append(index)
    return lines, fields


def recorded_version(root: pathlib.Path) -> str:
    """The version an unpacked gamfit sdist at ``root`` recorded in PKG-INFO when it was built.

    It must be the sdist's own release line or a development version of it, so a PKG-INFO this module
    did not write cannot name the engine.
    """
    pkg_info = root / "PKG-INFO"
    if not pkg_info.is_file():
        raise VersionError(
            f"{root} is neither a gam git checkout that tracks {PYPROJECT} nor an unpacked {DISTRIBUTION} "
            "sdist with a PKG-INFO, so nothing records the gamfit version of its engine"
        )
    lines, fields = _metadata_fields(pkg_info.read_text(encoding="utf-8"))
    names = [lines[index] for index in fields.get("Name", [])]
    if names != [f"Name: {DISTRIBUTION}"] or len(fields.get("Version", [])) != 1:
        raise VersionError(f"{pkg_info} does not name exactly one {DISTRIBUTION} version")
    version = lines[fields["Version"][0]][len("Version: "):]
    release = declared_version((root / PYPROJECT).read_bytes()) if (root / PYPROJECT).is_file() else None
    if release is None or not _RELEASE.fullmatch(release):
        raise VersionError(f"{root / PYPROJECT} declares no MAJOR.MINOR.PATCH release line")
    development = rf"{re.escape(_next_development(release))}\.dev(0|[1-9][0-9]*)\+g[0-9a-f]{{40}}(\.dirty)?"
    if version != release and not re.fullmatch(development, version):
        raise VersionError(
            f"{pkg_info} records {version!r}, which is neither the release line {release!r} nor a development "
            "version of it"
        )
    return version


def derive(root: pathlib.Path) -> tuple[str, str | None, bool | None]:
    """The gamfit version of the tree at ``root`` (see the module docstring), its commit and dirty state.

    For an unpacked sdist the commit and dirty state are None, as gam-build-identity records them there.
    """
    tracked = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "ls-files", "--error-unmatch", PYPROJECT],
        capture_output=True,
    )
    if tracked.returncode != 0:
        return recorded_version(root), None, None
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
    if not _RELEASE.fullmatch(release):
        raise VersionError(
            f"{PYPROJECT} at {commit} declares version {release!r}, not a MAJOR.MINOR.PATCH release line"
        )
    setters = release_commits(root, release)
    if setters == [commit] and not dirty:
        return release, commit, dirty
    distance = int(_git(root, "rev-list", "--count", commit, "--not", *setters))
    version = f"{_next_development(release)}.dev{distance}+g{commit}" + (".dirty" if dirty else "")
    return version, commit, dirty


def release(root: pathlib.Path) -> str:
    """The release version the tree at ``root`` is; refused for every other tree."""
    version, commit, dirty = derive(root)
    if not _RELEASE.fullmatch(version):
        raise VersionError(
            f"{commit or root}{' with uncommitted changes' if dirty else ''} is not the commit that set the "
            f"release line, so it builds as {version}; publish the commit that sets the new version instead"
        )
    return version


# ---------------------------------------------------------------------- stamp
def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}"


def _restamp_metadata(text: str, old: str, new: str) -> str:
    """A METADATA or PKG-INFO text with its one ``Version: old`` header line set to ``new``."""
    _, blank, body = text.partition("\n\n")
    lines, fields = _metadata_fields(text)
    names = [lines[index] for index in fields.get("Name", [])]
    versions = fields.get("Version", [])
    if names != [f"Name: {DISTRIBUTION}"] or len(versions) != 1 or lines[versions[0]] != f"Version: {old}":
        raise VersionError(f"the metadata does not name exactly {DISTRIBUTION} {old}")
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
    listed = [path for path in dist.files or [] if len(path.parts) == 2 and path.parts[1] == "METADATA"]
    if len(listed) != 1 or not listed[0].parts[0].endswith(".dist-info"):
        raise VersionError(f"the installed {DISTRIBUTION} has no single dist-info METADATA in its RECORD")
    old_dir = pathlib.Path(dist.locate_file(listed[0].parts[0]))
    return old_dir.parent / restamp_dist_info(old_dir.parent, old_dir.name, dist.version, version)


def restamp_dist_info(base: pathlib.Path, name: str, old: str, version: str) -> str:
    """Rename the dist-info directory ``base/name`` to ``version`` and return its new name.

    Its METADATA gets the new version. A RECORD, which an installed dist-info carries and a prepared one
    does not, gets every row under the directory renamed and rehashed, with paths relative to ``base``.
    """
    if old == version:
        return name
    new_name = f"{DISTRIBUTION}-{version}.dist-info"
    old_dir, new_dir = base / name, base / new_name
    restamped = _restamp_metadata((old_dir / "METADATA").read_text(encoding="utf-8"), old, version)
    rows = None
    if (old_dir / "RECORD").is_file():
        with (old_dir / "RECORD").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    os.rename(old_dir, new_dir)
    (new_dir / "METADATA").write_text(restamped, encoding="utf-8")
    if rows is None:
        return new_name
    old_prefix, new_prefix = f"{name}/", f"{new_name}/"
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
        data = (base / path).read_bytes()
        writer.writerow([path, _record_hash(data), len(data)])
    writer.writerow([record_path, "", ""])
    (new_dir / "RECORD").write_text(buffer.getvalue(), encoding="utf-8")
    return new_name


def stamp_sdist(sdist: pathlib.Path, version: str) -> pathlib.Path:
    """Rewrite a gamfit sdist to ``version`` and return its new path, which replaces ``sdist``.

    The file name, the one top-level directory and PKG-INFO carry the version, so a build from the
    unpacked sdist reads it back with ``recorded_version``.
    """
    prefix, suffix = f"{DISTRIBUTION}-", ".tar.gz"
    if not (sdist.name.startswith(prefix) and sdist.name.endswith(suffix)):
        raise VersionError(f"{sdist} is not a {DISTRIBUTION} sdist file name")
    old = sdist.name[len(prefix):-len(suffix)]
    if old == version:
        return sdist
    old_root, new_root = f"{DISTRIBUTION}-{old}", f"{DISTRIBUTION}-{version}"
    target = sdist.with_name(new_root + suffix)
    handle, scratch = tempfile.mkstemp(dir=sdist.parent, prefix=".stamp-", suffix=suffix)
    os.close(handle)
    try:
        stamped = False
        with tarfile.open(sdist, "r:gz") as source, tarfile.open(scratch, "w:gz") as out:
            for member in source.getmembers():
                if member.name != old_root and not member.name.startswith(old_root + "/"):
                    raise VersionError(f"{sdist} holds {member.name}, outside its top-level directory {old_root}")
                data = source.extractfile(member).read() if member.isfile() else None
                if member.name == f"{old_root}/PKG-INFO":
                    data = _restamp_metadata(data.decode("utf-8"), old, version).encode("utf-8")
                    stamped = True
                member.name = new_root + member.name[len(old_root):]
                if member.islnk() and member.linkname.startswith(old_root + "/"):
                    member.linkname = new_root + member.linkname[len(old_root):]
                if data is None:
                    out.addfile(member)
                else:
                    member.size = len(data)
                    out.addfile(member, io.BytesIO(data))
        if not stamped:
            raise VersionError(f"{sdist} has no {old_root}/PKG-INFO")
        os.replace(scratch, target)
    except BaseException:
        os.unlink(scratch)
        raise
    sdist.unlink()
    return target


# ------------------------------------------------------------- PEP 517 backend
# The hooks run with the source tree as the working directory (PEP 517).
def _maturin():
    import maturin

    return maturin


def _unchanged(root: pathlib.Path, identity: tuple[str, str | None, bool | None]) -> None:
    now = derive(root)
    if now != identity:
        raise VersionError(
            f"{root} derived {identity[0]} when the build started and {now[0]} after it: the tree changed "
            "during the build, so rebuild"
        )


def get_requires_for_build_wheel(config_settings=None):
    return _maturin().get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _maturin().get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _maturin().get_requires_for_build_sdist(config_settings)


def _prepare(hook, metadata_directory, config_settings) -> str:
    root = pathlib.Path.cwd()
    identity = derive(root)
    name = hook(metadata_directory, config_settings)
    _unchanged(root, identity)
    prefix, suffix = f"{DISTRIBUTION}-", ".dist-info"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise VersionError(f"maturin prepared {name}, not a {DISTRIBUTION} dist-info directory")
    return restamp_dist_info(pathlib.Path(metadata_directory), name, name[len(prefix):-len(suffix)], identity[0])


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _prepare(_maturin().prepare_metadata_for_build_wheel, metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _prepare(_maturin().prepare_metadata_for_build_editable, metadata_directory, config_settings)


def _build(hook, wheel_directory, config_settings, metadata_directory) -> str:
    root = pathlib.Path.cwd()
    identity = derive(root)
    # PEP 517 hands build_wheel the prepared .dist-info directory itself, not the directory holding it.
    if metadata_directory is not None:
        expected = f"{DISTRIBUTION}-{identity[0]}.dist-info"
        if pathlib.Path(metadata_directory).name != expected:
            raise VersionError(
                f"the metadata prepared as {metadata_directory} is not {expected}, the version {root} derives "
                "now: the tree changed between preparing the metadata and building the wheel, so rebuild"
            )
    # maturin is not handed the prepared directory: it would look in it for the dist-info under the
    # unstamped version it wrote there.
    name = hook(wheel_directory, config_settings, None)
    _unchanged(root, identity)
    return stamp_wheel(pathlib.Path(wheel_directory) / name, identity[0]).name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(_maturin().build_wheel, wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(_maturin().build_editable, wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    root = pathlib.Path.cwd()
    identity = derive(root)
    name = _maturin().build_sdist(sdist_directory, config_settings)
    _unchanged(root, identity)
    return stamp_sdist(pathlib.Path(sdist_directory) / name, identity[0]).name


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

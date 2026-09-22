#!/usr/bin/env python3
"""Install one built gamfit wheel into a fresh virtualenv and use it as a user would.

A wheel that builds is not a wheel that works. Missing package data, a
dependency the code imports but the metadata never declares, a dev extra that
only the repo checkout happened to provide, or an extension that will not load
on its platform all pass ``maturin build`` and fail on a user's first
``import gamfit``. This script is the installed-wheel gate (PKG-08 of the
pyGAM audit):

``run``
    Create an empty virtualenv beside no checkout, install the one wheel of
    the family built for its interpreter (the abi3 wheel on a GIL build, the
    interpreter's own ``cp3XYt`` wheel on a free-threaded one) plus the runtime
    requirements its own metadata declares (binaries only), and re-execute this
    file in ``check`` mode inside that venv under ``-I`` from a scratch
    directory, so neither the working tree nor ``PYTHONPATH`` nor user
    site-packages can stand in for the installed package.

    ``--resolution lowest`` installs the declared lower bound of every runtime
    requirement that has a binary for the interpreter, which is what makes the
    ``numpy>=`` floor in pyproject.toml a tested claim rather than a guess.

``check``
    Runs inside the fresh venv: the environment holds exactly gamfit and its
    declared requirement closure, the wheel carries only the package's own
    sources, typing marker and compiled extension, and a small gaussian and
    binomial fit, predict, summary and save/load round-trip all work. On a
    free-threaded interpreter the GIL must still be off after the import: an
    extension that does not declare free-threading support re-enables it.

``matrix``
    Print the GitHub Actions job matrix for the wheel families a workflow
    built. The Python versions come from the trove classifiers in
    pyproject.toml, so the versions the package advertises and the versions
    CI installs it on cannot drift apart.

``interpreters``
    Print the interpreters maturin builds wheels for, from the same
    classifiers: the requires-python floor for the abi3 wheel, then one
    free-threaded interpreter per advertised version that has one, because
    the stable ABI does not exist for free-threaded builds and each needs its
    own version-specific wheel.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

# One entry per wheel artifact pypi-wheels.yml uploads (`wheels-<family>`):
# the runner that can execute it, and for musl families the Alpine image the
# check runs in (a glibc runner cannot load a musllinux extension).
FAMILIES: dict[str, dict[str, object]] = {
    "linux-x86_64": {"runner": "ubuntu-24.04", "musl": False},
    "linux-aarch64": {"runner": "ubuntu-24.04-arm", "musl": False},
    "musllinux-x86_64": {"runner": "ubuntu-24.04", "musl": True},
    "musllinux-aarch64": {"runner": "ubuntu-24.04-arm", "musl": True},
    "macos-x86_64": {"runner": "macos-15-intel", "musl": False},
    "macos-aarch64": {"runner": "macos-14", "musl": False},
    "windows-x64": {"runner": "windows-latest", "musl": False},
}

# The pypi-wheels.yml `scope` input names which families a run builds.
SCOPES: dict[str, tuple[str, ...]] = {
    "x86": ("linux-x86_64",),
    "linux": ("linux-x86_64", "linux-aarch64", "musllinux-x86_64", "musllinux-aarch64"),
    "full": tuple(FAMILIES),
}

# Every advertised minor version runs on this family; the others run the
# floor and the newest version, which bracket the abi3 range the wheel serves.
REFERENCE_FAMILY = "linux-x86_64"

_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")
_FREE_THREADING = re.compile(r"^Programming Language :: Python :: Free Threading :: ")

# The first free-threaded CPython the extension can be built for. CPython has a
# free-threaded build (PEP 703) from 3.13, but PyO3 0.29 (Cargo.lock) supports
# it only from 3.14: its build script stops with "PyO3 does not support the
# free-threaded build of CPython versions below 3.14".
FREE_THREADED_SINCE = "3.14"


# ---------------------------------------------------------------------- matrix
def _pyproject() -> dict:
    import tomllib

    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def classifier_python_versions(project: dict) -> list[str]:
    """The ``3.X`` versions the trove classifiers advertise, oldest first."""
    found = [
        match.group(1)
        for classifier in project["classifiers"]
        if (match := _CLASSIFIER.match(classifier))
    ]
    return sorted(found, key=_version_key)


def free_threaded_python_versions(project: dict) -> list[str]:
    """The free-threaded interpreters (``3.14t`` ...) the package advertises, oldest first.

    A Free Threading classifier claims support on every advertised version
    the extension can be built free-threaded for; without one there are none.
    """
    if not any(_FREE_THREADING.match(classifier) for classifier in project["classifiers"]):
        return []
    since = _version_key(FREE_THREADED_SINCE)
    return [f"{v}t" for v in classifier_python_versions(project) if _version_key(v) >= since]


def build_interpreters(project: dict) -> list[str]:
    """The interpreters maturin builds for: abi3 on the floor, then each free-threaded one."""
    return [requires_python_floor(project), *free_threaded_python_versions(project)]


def requires_python_floor(project: dict) -> str:
    spec = project["requires-python"].strip()
    match = re.fullmatch(r">=\s*(3\.\d+)", spec)
    if match is None:
        raise ValueError(f"requires-python must be one '>=3.X' floor, got {spec!r}")
    return match.group(1)


def build_matrix(families: list[str], project: dict) -> dict[str, list[dict[str, object]]]:
    versions = classifier_python_versions(project)
    if not versions:
        raise ValueError("pyproject.toml advertises no 'Programming Language :: Python :: 3.X'")
    floor, newest = versions[0], versions[-1]
    if floor != requires_python_floor(project):
        raise ValueError(
            f"oldest classifier {floor} disagrees with requires-python {project['requires-python']!r}"
        )
    include: list[dict[str, object]] = []
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"unknown wheel family {family!r}; known: {sorted(FAMILIES)}")
        lanes = versions if family == REFERENCE_FAMILY else sorted({floor, newest}, key=_version_key)
        for python in lanes:
            include.append(
                {
                    "family": family,
                    "runner": FAMILIES[family]["runner"],
                    "musl": FAMILIES[family]["musl"],
                    "python": python,
                    # The floor interpreter gets the lowest declared requirement
                    # versions; every other lane gets the newest.
                    "resolution": "lowest" if python == floor else "highest",
                    "free_threaded": False,
                }
            )
        # Each free-threaded wheel is its own binary, so every one runs on
        # every family. The requirement floor is the floor lane's to test (it
        # predates free-threaded wheels: numpy 1.26 has none).
        for python in free_threaded_python_versions(project):
            include.append(
                {
                    "family": family,
                    "runner": FAMILIES[family]["runner"],
                    "musl": FAMILIES[family]["musl"],
                    "python": python,
                    "resolution": "highest",
                    "free_threaded": True,
                }
            )
    return {"include": include}


# --------------------------------------------------------------------- metadata
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_requirements(requires_dist: list[str] | None) -> set[str]:
    """Names of the requirements a plain ``pip install gamfit`` pulls in.

    A requirement whose marker mentions ``extra`` belongs to an optional extra.
    Any other marker is refused rather than guessed at: evaluating markers
    needs ``packaging``, which a fresh venv does not have, and gamfit's
    runtime requirements are unconditional.
    """
    names: set[str] = set()
    for line in requires_dist or []:
        requirement, _, marker = line.partition(";")
        if re.search(r"\bextra\b", marker):
            continue
        if marker.strip():
            raise ValueError(f"runtime requirement with an environment marker: {line!r}")
        match = _REQUIREMENT_NAME.match(requirement)
        if match is None:
            raise ValueError(f"unparseable Requires-Dist entry {line!r}")
        names.add(canonical_name(match.group(1)))
    return names


def requirement_closure(root: str, requires_of) -> set[str]:
    """``root`` plus every distribution its runtime requirements reach."""
    seen: set[str] = set()
    pending = [canonical_name(root)]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(runtime_requirements(requires_of(name)) - seen)
    return seen


def wheel_for(wheels: list[pathlib.Path], python_tag: str, free_threaded: bool) -> pathlib.Path:
    """The one wheel of a family an interpreter installs.

    A GIL interpreter takes the abi3 wheel; a free-threaded one takes the
    wheel built for its own ABI (``cp314t``), which a GIL build cannot load.
    """
    abi = f"{python_tag}t" if free_threaded else "abi3"
    # name-version[-build]-python-abi-platform.whl
    matches = [wheel for wheel in wheels if wheel.name.removesuffix(".whl").split("-")[-2] == abi]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {abi} gamfit wheel among {[w.name for w in wheels]}")
    return matches[0]


def stray_wheel_files(paths: list[str], dist_info: str, extension_suffixes: list[str]) -> list[str]:
    """Files the installed wheel carries that are not part of the package.

    The package is its Python sources, the PEP 561 marker and one compiled
    ``_rust`` extension; a repaired wheel also carries the shared libraries
    auditwheel vendored beside it, under ``gamfit.libs/`` (the musllinux
    wheels ship libgcc_s there, which Alpine does not guarantee); everything
    else must live in the dist-info directory. Test data, Rust sources,
    notebooks or build leftovers land here.
    """
    stray: list[str] = []
    for path in paths:
        parts = pathlib.PurePosixPath(path).parts
        if parts[0] == dist_info:
            continue
        if parts[0] == "gamfit.libs" and len(parts) == 2 and ".so" in parts[1].split("-")[-1]:
            continue
        if parts[0] != "gamfit":
            stray.append(path)
            continue
        name = parts[-1]
        if "__pycache__" in parts and name.endswith(".pyc"):
            continue  # written by an installer that byte-compiles, not shipped
        if name.endswith((".py", ".pyi")) or path == "gamfit/py.typed":
            continue
        if len(parts) == 2 and any(name == "_rust" + suffix for suffix in extension_suffixes):
            continue
        stray.append(path)
    return stray


# ------------------------------------------------------------------------ check
def _gil_enabled() -> bool:
    if sys.version_info >= (3, 13):
        return sys._is_gil_enabled()
    return True


def check() -> None:
    import importlib.machinery
    from importlib import metadata

    if not sys.flags.isolated:
        raise SystemExit("check must run under `python -I` so no checkout or PYTHONPATH is importable")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("check must run inside the fresh virtualenv, not the base interpreter")

    installed = {canonical_name(dist.metadata["Name"]) for dist in metadata.distributions()}
    expected = requirement_closure("gamfit", metadata.requires)
    if installed != expected:
        raise SystemExit(
            f"fresh venv holds {sorted(installed)}; gamfit's declared runtime closure is {sorted(expected)}"
        )

    dist = metadata.distribution("gamfit")
    files = [str(path) for path in dist.files or []]
    dist_info = next(
        (pathlib.PurePosixPath(p).parts[0] for p in files if p.endswith(".dist-info/METADATA")), None
    )
    if dist_info is None:
        raise SystemExit("installed gamfit has no RECORD entry for its METADATA")
    stray = stray_wheel_files(files, dist_info, importlib.machinery.EXTENSION_SUFFIXES)
    if stray:
        raise SystemExit("wheel ships files outside the package:\n  " + "\n  ".join(stray))
    if "gamfit/py.typed" not in files:
        raise SystemExit("wheel is missing the PEP 561 gamfit/py.typed marker")

    import sysconfig

    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

    import numpy as np

    if free_threaded and _gil_enabled():
        raise SystemExit(f"importing numpy {np.__version__} re-enabled the GIL")

    import gamfit

    if free_threaded and _gil_enabled():
        raise SystemExit("importing gamfit re-enabled the GIL: its extension does not declare free-threading support")

    site = pathlib.Path(sys.prefix).resolve()
    package = pathlib.Path(gamfit.__file__).resolve()
    if site not in package.parents:
        raise SystemExit(f"imported gamfit from {package}, not from the fresh venv {site}")
    if gamfit.__version__ != dist.version:
        raise SystemExit(f"gamfit.__version__ {gamfit.__version__!r} != metadata {dist.version!r}")

    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    signal = np.sin(6.0 * x)
    new = {"x": np.linspace(0.05, 0.95, 7)}

    gaussian = gamfit.fit({"x": x, "y": signal + rng.normal(0.0, 0.3, n)}, "y ~ s(x)")
    mean = np.asarray(gaussian.predict(new), dtype=float)
    if mean.shape != (7,) or not np.all(np.isfinite(mean)):
        raise SystemExit(f"gaussian predict returned {mean!r}")
    if not str(gaussian.summary()).strip():
        raise SystemExit("gaussian summary rendered empty")

    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / "model.gam"
        gaussian.save(path)
        reloaded = gamfit.load(path)
    if not np.array_equal(np.asarray(reloaded.predict(new), dtype=float), mean):
        raise SystemExit("save/load round-trip changed the gaussian predictions")

    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-3.0 * signal))).astype(float)
    binomial = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="binomial")
    prob = np.asarray(binomial.predict(new), dtype=float)
    if prob.shape != (7,) or not np.all((prob > 0.0) & (prob < 1.0)):
        raise SystemExit(f"binomial predict returned {prob!r}")

    build = "free-threaded" if free_threaded else "GIL"
    print(
        f"gamfit {gamfit.__version__} OK on {build} Python {sys.version.split()[0]} "
        f"({sys.platform}) with numpy {np.__version__}; installed: {sorted(installed)}"
    )


# -------------------------------------------------------------------------- run
def run(wheel_dir: pathlib.Path, resolution: str, interpreter: str) -> None:
    wheels = sorted(path.resolve() for path in wheel_dir.glob("gamfit-*.whl"))
    uv = [sys.executable, "-m", "uv"]
    # uv must use the interpreter under test and PyPI only: no managed-Python
    # download, no uv.toml or pyproject picked up from wherever the job
    # happens to be, and no PYTHON_GIL override of a free-threaded build.
    env = {**os.environ, "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1"}
    for name in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHON_GIL"):
        env.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="gamfit-wheel-smoke-") as scratch:
        scratch_path = pathlib.Path(scratch)
        venv = scratch_path / "venv"
        subprocess.run([*uv, "venv", "--python", interpreter, str(venv)], check=True, env=env, cwd=scratch)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        probe = subprocess.run(
            [
                str(python), "-I", "-c",
                "import sys, sysconfig; "
                "print(f'cp{sys.version_info[0]}{sys.version_info[1]}', bool(sysconfig.get_config_var('Py_GIL_DISABLED')))",
            ],
            check=True, env=env, cwd=scratch, capture_output=True, text=True,
        )
        python_tag, free_threaded = probe.stdout.split()
        wheel = wheel_for(wheels, python_tag, free_threaded == "True")
        subprocess.run(
            [
                *uv, "pip", "install",
                "--python", str(python),
                "--only-binary", ":all:",
                "--resolution", resolution,
                str(wheel),
            ],
            check=True,
            env=env,
            cwd=scratch,
        )
        # A copy in the scratch directory: nothing next to the checked-in
        # script (the repo root is two levels up) can be on the check's path.
        script = scratch_path / "wheel_smoke_check.py"
        shutil.copyfile(__file__, script)
        subprocess.run([str(python), "-I", str(script), "check"], check=True, env=env, cwd=scratch)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="install one wheel into a fresh venv and check it")
    run_parser.add_argument("--wheel-dir", type=pathlib.Path, required=True)
    run_parser.add_argument("--resolution", choices=("lowest", "highest"), required=True)
    run_parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter the venv is built on (a version uv can find, or a path); default: this one",
    )
    sub.add_parser("check", help="(inside the fresh venv) verify the installed wheel")
    matrix_parser = sub.add_parser("matrix", help="print the smoke-test job matrix as JSON")
    selector = matrix_parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--scope", choices=sorted(SCOPES))
    selector.add_argument("--families", nargs="+", choices=sorted(FAMILIES))
    sub.add_parser("interpreters", help="print the Python versions maturin builds wheels for")
    args = parser.parse_args(argv)

    if args.command == "run":
        run(args.wheel_dir, args.resolution, args.python)
    elif args.command == "check":
        check()
    elif args.command == "interpreters":
        print(" ".join(build_interpreters(_pyproject()["project"])))
    else:
        families = list(SCOPES[args.scope]) if args.scope else args.families
        print(json.dumps(build_matrix(families, _pyproject()["project"]), separators=(",", ":")))


if __name__ == "__main__":
    main()

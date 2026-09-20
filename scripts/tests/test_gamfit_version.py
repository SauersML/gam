"""Contracts for the build-derived gamfit version (gam#3157), on throwaway git repositories and wheels."""

import base64
import csv
import hashlib
import importlib.util
import io
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
import unittest.mock
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/gamfit_version.py"
SPEC = importlib.util.spec_from_file_location("gamfit_version", SCRIPT)
gamfit_version = importlib.util.module_from_spec(SPEC)
sys.modules["gamfit_version"] = gamfit_version
SPEC.loader.exec_module(gamfit_version)

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def pyproject(version: str, note: str = "") -> str:
    """The repository's layout: a dynamic [project] version and the release line under [tool.gamfit]."""
    return (
        '[build-system]\nrequires = ["maturin>=1.7,<2"]\nbuild-backend = "gamfit_version"\n'
        'backend-path = ["scripts"]\n\n[project]\nname = "gamfit"\ndynamic = ["version"]\n'
        f'description = "d"\nrequires-python = ">=3.10"\n\n[tool.gamfit]\nversion = "{version}"\n\n'
        f'[tool.maturin]\nmodule-name = "gamfit._rust"\n# {note}\n'
    )


def static_pyproject(version: str) -> str:
    """The layout before gam#3157's backend: the release line as the static [project] version."""
    return f'[project]\nname = "gamfit"\nversion = "{version}"\ndescription = "d"\n'


class Repo:
    def __init__(self, path: pathlib.Path):
        self.path = path
        path.mkdir()
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-C", str(self.path), *args],
            capture_output=True, text=True, env=GIT_ENV, check=True,
        )
        return done.stdout.strip()

    def commit(self, files: dict[str, str], message: str) -> str:
        for name, text in files.items():
            (self.path / name).write_text(text)
            self.git("add", name)
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def version(self) -> str:
        return gamfit_version.derive(self.path)[0]


class DeriveTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.repo = Repo(pathlib.Path(self.scratch.name) / "gam")
        self.base = self.repo.commit({"pyproject.toml": pyproject("2.3.4"), "engine.rs": "a\n"}, "release: 2.3.4")
        self.release = self.repo.commit({"pyproject.toml": pyproject("2.3.5")}, "release: 2.3.5")

    def tearDown(self):
        self.scratch.cleanup()

    def test_the_commit_that_sets_the_release_line_is_the_release(self):
        self.assertEqual(self.repo.version(), "2.3.5")

    def test_every_later_engine_has_its_own_development_version_naming_its_commit(self):
        engine = self.repo.commit({"engine.rs": "b\n"}, "engine change")
        first = self.repo.version()
        self.assertEqual(first, f"2.3.6.dev1+g{engine}")
        # Editing pyproject.toml without moving the release line does not make a new release.
        packaging = self.repo.commit({"pyproject.toml": pyproject("2.3.5", "reworded")}, "packaging edit")
        second = self.repo.version()
        self.assertEqual(second, f"2.3.6.dev2+g{packaging}")
        # Positive control for the defect: the static line says 2.3.5 at all three commits.
        for commit in (self.release, engine, packaging):
            self.assertEqual(gamfit_version._version_at(self.repo.path, commit), "2.3.5")
        self.assertEqual(len({"2.3.5", first, second}), 3)
        self.assertNotEqual(first.partition("+")[0], second.partition("+")[0])

    def test_a_dirty_tree_is_never_the_release(self):
        (self.repo.path / "engine.rs").write_text("edited\n")
        self.assertEqual(self.repo.version(), f"2.3.6.dev0+g{self.release}.dirty")

    def test_an_untracked_file_does_not_make_the_tree_dirty(self):
        (self.repo.path / "notes.txt").write_text("scratch\n")
        self.assertEqual(self.repo.version(), "2.3.5")

    def test_a_merge_that_keeps_one_sides_release_line_is_not_a_release(self):
        # A side branch from before the release edits another pyproject.toml line. The merge keeps
        # main's 2.3.5, so neither the merge nor the side commit set it: the release is still `release`.
        self.repo.git("checkout", "-q", "-b", "side", self.base)
        side = self.repo.commit({"pyproject.toml": pyproject("2.3.4") + "# side\n"}, "side edit")
        self.repo.git("checkout", "-q", "main")
        self.repo.git("merge", "-q", "--no-edit", "side")
        merge = self.repo.git("rev-parse", "HEAD")
        self.assertEqual(gamfit_version._version_at(self.repo.path, merge), "2.3.5")
        self.assertEqual(gamfit_version.release_commits(self.repo.path, "2.3.5"), [self.release])
        # The merge and the side commit are the two commits the release does not reach.
        self.assertEqual(self.repo.version(), f"2.3.6.dev2+g{merge}")
        self.assertNotEqual(side, merge)

    def test_a_shallow_clone_is_refused(self):
        self.repo.commit({"engine.rs": "b\n"}, "engine change")
        shallow = pathlib.Path(self.scratch.name) / "shallow"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", self.repo.path.as_uri(), str(shallow)],
            check=True, env=GIT_ENV, capture_output=True,
        )
        with self.assertRaisesRegex(gamfit_version.VersionError, "shallow clone"):
            gamfit_version.derive(shallow)

    def test_a_tree_outside_a_checkout_is_refused(self):
        loose = pathlib.Path(self.scratch.name) / "loose"
        loose.mkdir()
        (loose / "pyproject.toml").write_text(pyproject("2.3.5"))
        with self.assertRaisesRegex(gamfit_version.VersionError, "neither a gam git checkout"):
            gamfit_version.derive(loose)

    def test_moving_the_release_line_out_of_project_does_not_set_it(self):
        # The layout change that made [project] version dynamic keeps the value, so it releases nothing.
        repo = Repo(pathlib.Path(self.scratch.name) / "static")
        release = repo.commit({"pyproject.toml": static_pyproject("2.3.5"), "engine.rs": "a\n"}, "release: 2.3.5")
        self.assertEqual(repo.version(), "2.3.5")
        moved = repo.commit({"pyproject.toml": pyproject("2.3.5")}, "derive the version in the backend")
        self.assertEqual(gamfit_version.release_commits(repo.path, "2.3.5"), [release])
        self.assertEqual(repo.version(), f"2.3.6.dev1+g{moved}")

    def test_only_the_commit_that_sets_the_release_line_can_be_published_as_it(self):
        self.assertEqual(gamfit_version.release(self.repo.path), "2.3.5")
        self.repo.commit({"engine.rs": "b\n"}, "engine change")
        with self.assertRaisesRegex(gamfit_version.VersionError, "not the commit that set the release line"):
            gamfit_version.release(self.repo.path)
        self.repo.commit({"pyproject.toml": pyproject("2.3.6")}, "release: 2.3.6")
        self.assertEqual(gamfit_version.release(self.repo.path), "2.3.6")
        (self.repo.path / "engine.rs").write_text("edited\n")
        with self.assertRaisesRegex(gamfit_version.VersionError, "with uncommitted changes"):
            gamfit_version.release(self.repo.path)

    def test_a_release_line_that_is_not_major_minor_patch_is_refused(self):
        self.repo.commit({"pyproject.toml": pyproject("2.3.6rc1")}, "not a release line")
        with self.assertRaisesRegex(gamfit_version.VersionError, "MAJOR.MINOR.PATCH"):
            self.repo.version()


class ReleaseLineTests(unittest.TestCase):
    def test_the_release_line_is_the_first_version_line_as_build_rs_reads_it(self):
        text = b'[build-system]\nrequires = ["maturin"]\n\n[project]\nname = "gamfit"\nversion = "2.3.5"\n'
        self.assertEqual(gamfit_version.declared_version(text), "2.3.5")
        self.assertEqual(gamfit_version.declared_version(text + b'[tool.x]\nversion = "9.9.9"\n'), "2.3.5")
        self.assertIsNone(gamfit_version.declared_version(b'[project]\nname = "gamfit"\n'))

    def test_the_repository_release_line_is_the_one_gam_pyffi_carries(self):
        # build.rs holds crates/gam-pyffi/Cargo.toml's first version line to pyproject.toml's, and maturin
        # gives the dynamic [project] version from that Cargo.toml.
        project = gamfit_version.declared_version((ROOT / "pyproject.toml").read_bytes())
        pyffi = gamfit_version.declared_version((ROOT / "crates/gam-pyffi/Cargo.toml").read_bytes())
        self.assertIsNotNone(project)
        self.assertEqual(project, pyffi)

    def test_the_repository_builds_through_this_backend_and_ships_it_in_the_sdist(self):
        import tomllib

        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["build-system"]["build-backend"], SCRIPT.stem)
        self.assertEqual([ROOT / path for path in config["build-system"]["backend-path"]], [SCRIPT.parent])
        # PEP 621: a backend that computes the version must not also declare it statically.
        self.assertNotIn("version", config["project"])
        self.assertIn("version", config["project"]["dynamic"])
        self.assertEqual(config["tool"]["gamfit"]["version"], gamfit_version.declared_version(
            (ROOT / "pyproject.toml").read_bytes()))
        shipped = [entry for entry in config["tool"]["maturin"]["include"] if entry.get("format") == "sdist"]
        self.assertIn(SCRIPT.relative_to(ROOT).as_posix(), [entry["path"] for entry in shipped])


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build_wheel(directory: pathlib.Path, version: str, package: str) -> pathlib.Path:
    """A py3-none-any gamfit wheel shaped like maturin's: package files, then a dist-info with RECORD."""
    dist_info = f"gamfit-{version}.dist-info"
    files = {
        "gamfit/__init__.py": package.encode(),
        "gamfit/_rust.abi3.so": b"\x7fELF not really",
        f"{dist_info}/METADATA": f"Metadata-Version: 2.4\nName: gamfit\nVersion: {version}\n\nbody\n".encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: py3-none-any\n",
    }
    record = "".join(f"{path},{record_hash(data)},{len(data)}\n" for path, data in files.items())
    record += f"{dist_info}/RECORD,,\n"
    wheel = directory / f"gamfit-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for path, data in files.items():
            info = zipfile.ZipInfo(path, date_time=(2026, 9, 19, 0, 0, 0))
            info.external_attr = (0o755 if path.endswith(".so") else 0o644) << 16
            out.writestr(info, data)
        out.writestr(f"{dist_info}/RECORD", record)
    return wheel


def engine_package(commit: str, dirty: bool) -> str:
    return (
        "def build_info():\n"
        f"    return {{'available': True, 'commit': {commit!r}, 'dirty': {dirty!r}}}\n"
    )


class StampTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.scratch.name)
        self.version = "2.3.6.dev3+g" + "a" * 40

    def tearDown(self):
        self.scratch.cleanup()

    def test_a_stamped_wheel_carries_the_version_everywhere_and_a_valid_record(self):
        wheel = build_wheel(self.dir, "2.3.5", "")
        stamped = gamfit_version.stamp_wheel(wheel, self.version)
        self.assertEqual(stamped.name, f"gamfit-{self.version}-py3-none-any.whl")
        self.assertFalse(wheel.exists())
        dist_info = f"gamfit-{self.version}.dist-info"
        with zipfile.ZipFile(stamped) as archive:
            names = archive.namelist()
            self.assertFalse([name for name in names if "2.3.5" in name], names)
            self.assertEqual(names[-1], f"{dist_info}/RECORD")
            metadata = archive.read(f"{dist_info}/METADATA").decode()
            self.assertIn(f"\nVersion: {self.version}\n", metadata)
            self.assertNotIn("2.3.5", metadata)
            self.assertEqual(archive.getinfo("gamfit/_rust.abi3.so").external_attr >> 16, 0o755)
            rows = list(csv.reader(io.StringIO(archive.read(f"{dist_info}/RECORD").decode())))
            self.assertEqual(sorted(row[0] for row in rows), sorted(names))
            for path, digest, size in rows:
                if path.endswith("/RECORD"):
                    self.assertEqual((digest, size), ("", ""))
                    continue
                data = archive.read(path)
                self.assertEqual((digest, int(size)), (record_hash(data), len(data)), path)

    def test_a_stamped_wheel_installs_under_its_new_version(self):
        stamped = gamfit_version.stamp_wheel(build_wheel(self.dir, "2.3.5", ""), self.version)
        target = self.dir / "site"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
             "--disable-pip-version-check", "--target", str(target), str(stamped)],
            check=True, capture_output=True,
        )
        from importlib import metadata

        [dist] = [dist for dist in metadata.distributions(path=[str(target)]) if dist.metadata["Name"] == "gamfit"]
        self.assertEqual(dist.version, self.version)

    def test_a_wheel_whose_metadata_disagrees_with_its_name_is_refused(self):
        wheel = build_wheel(self.dir, "2.3.5", "")
        wheel.rename(self.dir / "gamfit-2.3.4-py3-none-any.whl")
        with self.assertRaisesRegex(gamfit_version.VersionError, "no gamfit-2.3.4.dist-info/METADATA"):
            gamfit_version.stamp_wheel(self.dir / "gamfit-2.3.4-py3-none-any.whl", self.version)

    def test_the_release_wheel_is_left_as_built(self):
        wheel = build_wheel(self.dir, "2.3.5", "")
        before = wheel.read_bytes()
        self.assertEqual(gamfit_version.stamp_wheel(wheel, "2.3.5"), wheel)
        self.assertEqual(wheel.read_bytes(), before)


class StampInstalledTests(unittest.TestCase):
    """``stamp-installed`` runs in a child interpreter whose only gamfit is one installed to a target dir."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.scratch.name)
        self.repo = Repo(self.dir / "gam")
        self.repo.commit({"pyproject.toml": pyproject("2.3.5"), "engine.rs": "a\n"}, "release: 2.3.5")
        self.head = self.repo.commit({"engine.rs": "b\n"}, "engine change")

    def tearDown(self):
        self.scratch.cleanup()

    def install(self, commit: str, dirty: bool) -> pathlib.Path:
        wheels = self.dir / f"wheels-{commit[:8]}-{dirty}"
        wheels.mkdir()
        target = self.dir / f"site-{commit[:8]}-{dirty}"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index",
             "--disable-pip-version-check", "--target", str(target),
             str(build_wheel(wheels, "2.3.5", engine_package(commit, dirty)))],
            check=True, capture_output=True,
        )
        return target

    def stamp_installed(self, target: pathlib.Path) -> subprocess.CompletedProcess:
        # PYTHONPATH puts the target ahead of site-packages, so its gamfit is the one imported and the
        # first distribution named gamfit that importlib.metadata finds.
        env = {**os.environ, "PYTHONPATH": str(target), "PYTHONNOUSERSITE": "1"}
        return subprocess.run(
            [sys.executable, str(SCRIPT), "stamp-installed", "--root", str(self.repo.path)],
            capture_output=True, text=True, env=env,
        )

    def installed_version(self, target: pathlib.Path) -> str:
        from importlib import metadata

        [dist] = [dist for dist in metadata.distributions(path=[str(target)]) if dist.metadata["Name"] == "gamfit"]
        return dist.version

    def test_the_installed_engine_is_restamped_to_the_trees_version(self):
        target = self.install(self.head, False)
        old_dist_info = "gamfit-2.3.5.dist-info"
        with (target / old_dist_info / "RECORD").open(newline="") as handle:
            before = list(csv.reader(handle))
        done = self.stamp_installed(target)
        self.assertEqual(done.returncode, 0, done.stderr)
        expected = f"2.3.6.dev1+g{self.head}"
        self.assertEqual(self.installed_version(target), expected)
        dist_info = f"gamfit-{expected}.dist-info"
        self.assertFalse((target / old_dist_info).exists())
        with (target / dist_info / "RECORD").open(newline="") as handle:
            after = list(csv.reader(handle))
        # Rows outside the dist-info (the package files, which pip may list unhashed) are kept as pip wrote
        # them; every dist-info file is listed once under its new name with its current hash and size.
        self.assertEqual(
            [row for row in after if not row[0].startswith(dist_info)],
            [row for row in before if not row[0].startswith(old_dist_info)],
        )
        renamed = [row for row in after if row[0].startswith(dist_info)]
        self.assertEqual(
            sorted(row[0] for row in renamed),
            sorted(dist_info + row[0][len(old_dist_info):] for row in before if row[0].startswith(old_dist_info)),
        )
        for path, digest, size in renamed:
            if path == f"{dist_info}/RECORD":
                self.assertEqual((digest, size), ("", ""))
                continue
            data = (target / path).read_bytes()
            self.assertEqual((digest, int(size)), (record_hash(data), len(data)), path)

    def test_an_engine_built_from_another_commit_is_refused_and_left_alone(self):
        target = self.install("b" * 40, False)
        done = self.stamp_installed(target)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("the tree changed during the build", done.stderr)
        self.assertEqual(self.installed_version(target), "2.3.5")

    def test_an_engine_whose_dirty_state_differs_is_refused(self):
        target = self.install(self.head, True)
        done = self.stamp_installed(target)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("the tree changed during the build", done.stderr)


class FakeMaturin:
    """maturin's PEP 517 hooks as the backend sees them: artifacts named by the unstamped version.

    maturin takes that version from gam-pyffi's Cargo.toml, which build.rs holds to the release line, so
    the fake reads the release line of the tree it builds (the working directory, as under PEP 517).
    ``during_build`` runs inside a build hook, to change the tree while maturin works.
    """

    def __init__(self):
        self.during_build = None
        self.metadata_directories = []

    @staticmethod
    def version() -> str:
        return gamfit_version.declared_version(pathlib.Path("pyproject.toml").read_bytes())

    def get_requires_for_build_wheel(self, config_settings=None):
        return []

    get_requires_for_build_editable = get_requires_for_build_sdist = get_requires_for_build_wheel

    def prepare_metadata_for_build_wheel(self, metadata_directory, config_settings=None):
        name = f"gamfit-{self.version()}.dist-info"
        (pathlib.Path(metadata_directory) / name).mkdir()
        (pathlib.Path(metadata_directory) / name / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: gamfit\nVersion: {self.version()}\n\nbody\n"
        )
        return name

    prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel

    def build_wheel(self, wheel_directory, config_settings=None, metadata_directory=None):
        self.metadata_directories.append(metadata_directory)
        if self.during_build is not None:
            self.during_build()
        return build_wheel(pathlib.Path(wheel_directory), self.version(), "").name

    build_editable = build_wheel

    def build_sdist(self, sdist_directory, config_settings=None):
        root = f"gamfit-{self.version()}"
        members = {
            "PKG-INFO": f"Metadata-Version: 2.4\nName: gamfit\nVersion: {self.version()}\n\nbody\n".encode(),
            "pyproject.toml": pathlib.Path("pyproject.toml").read_bytes(),
            "engine.rs": pathlib.Path("engine.rs").read_bytes(),
        }
        sdist = pathlib.Path(sdist_directory) / f"{root}.tar.gz"
        with tarfile.open(sdist, "w:gz") as out:
            for name, data in members.items():
                info = tarfile.TarInfo(f"{root}/{name}")
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
        return sdist.name


class BackendTests(unittest.TestCase):
    """The PEP 517 hooks on a git checkout and on sdists cut at a release and at a later commit."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.scratch.name)
        self.repo = Repo(self.dir / "gam")
        self.repo.commit({"pyproject.toml": pyproject("2.3.4"), "engine.rs": "a\n"}, "release: 2.3.4")
        self.release = self.repo.commit({"pyproject.toml": pyproject("2.3.5")}, "release: 2.3.5")
        self.maturin = FakeMaturin()
        patcher = unittest.mock.patch.dict(sys.modules, {"maturin": self.maturin})
        patcher.start()
        # Cleanups run last-in first-out: leave the scratch directory before deleting it.
        self.addCleanup(patcher.stop)
        self.addCleanup(self.scratch.cleanup)
        self.addCleanup(os.chdir, os.getcwd())

    def out(self, name: str) -> pathlib.Path:
        path = self.dir / name
        path.mkdir()
        return path

    def wheel_version(self, wheel: pathlib.Path) -> str:
        with zipfile.ZipFile(wheel) as archive:
            [metadata] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            lines = archive.read(metadata).decode().split("\n")
        [version] = [line[len("Version: "):] for line in lines if line.startswith("Version: ")]
        self.assertEqual(metadata, f"gamfit-{version}.dist-info/METADATA")
        self.assertEqual(wheel.name, f"gamfit-{version}-py3-none-any.whl")
        return version

    def sdist(self) -> tuple[pathlib.Path, pathlib.Path]:
        """Build an sdist of the checkout and unpack it outside any git repository."""
        os.chdir(self.repo.path)
        sdist = self.out("sdist") / gamfit_version.build_sdist(str(self.dir / "sdist"))
        unpacked = self.out("unpacked")
        with tarfile.open(sdist) as archive:
            archive.extractall(unpacked, filter="data")
        [tree] = list(unpacked.iterdir())
        return sdist, tree

    def test_a_checkout_prepares_and_builds_its_derived_version(self):
        engine = self.repo.commit({"engine.rs": "b\n"}, "engine change")
        expected = f"2.3.6.dev1+g{engine}"
        os.chdir(self.repo.path)
        metadata = self.out("metadata")
        name = gamfit_version.prepare_metadata_for_build_wheel(str(metadata))
        self.assertEqual(name, f"gamfit-{expected}.dist-info")
        self.assertIn(f"\nVersion: {expected}\n", (metadata / name / "METADATA").read_text())
        wheel = self.out("wheel") / gamfit_version.build_wheel(str(self.dir / "wheel"), None, str(metadata / name))
        self.assertEqual(self.wheel_version(wheel), expected)
        editable = self.out("editable") / gamfit_version.build_editable(str(self.dir / "editable"))
        self.assertEqual(self.wheel_version(editable), expected)
        # maturin never sees the stamped metadata directory, whose name it did not write.
        self.assertEqual(self.maturin.metadata_directories, [None, None])

    def test_an_sdist_cut_at_the_release_commit_builds_the_release(self):
        sdist, tree = self.sdist()
        self.assertEqual(sdist.name, "gamfit-2.3.5.tar.gz")
        self.assertEqual(tree.name, "gamfit-2.3.5")
        os.chdir(tree)
        self.assertEqual(gamfit_version.derive(tree), ("2.3.5", None, None))
        wheel = self.out("wheel") / gamfit_version.build_wheel(str(self.dir / "wheel"))
        self.assertEqual(self.wheel_version(wheel), "2.3.5")

    def test_an_sdist_cut_at_a_later_commit_carries_its_development_version(self):
        engine = self.repo.commit({"engine.rs": "b\n"}, "engine change")
        expected = f"2.3.6.dev1+g{engine}"
        sdist, tree = self.sdist()
        self.assertEqual(sdist.name, f"gamfit-{expected}.tar.gz")
        self.assertEqual(tree.name, f"gamfit-{expected}")
        self.assertIn(f"\nVersion: {expected}\n", (tree / "PKG-INFO").read_text())
        # Positive control: the sdist's own pyproject.toml still names the release line.
        self.assertEqual(gamfit_version.declared_version((tree / "pyproject.toml").read_bytes()), "2.3.5")
        os.chdir(tree)
        metadata = self.out("metadata")
        name = gamfit_version.prepare_metadata_for_build_wheel(str(metadata))
        self.assertEqual(name, f"gamfit-{expected}.dist-info")
        wheel = self.out("wheel") / gamfit_version.build_wheel(str(self.dir / "wheel"), None, str(metadata / name))
        self.assertEqual(self.wheel_version(wheel), expected)

    def test_an_sdist_whose_pkg_info_names_another_version_is_refused(self):
        sdist, tree = self.sdist()
        pkg_info = tree / "PKG-INFO"
        pkg_info.write_text(pkg_info.read_text().replace("Version: 2.3.5", "Version: 9.9.9"))
        os.chdir(tree)
        with self.assertRaisesRegex(gamfit_version.VersionError, "neither the release line"):
            gamfit_version.build_wheel(str(self.out("wheel")))

    def test_a_tree_that_moves_between_preparing_and_building_is_refused(self):
        os.chdir(self.repo.path)
        metadata = self.out("metadata")
        name = gamfit_version.prepare_metadata_for_build_wheel(str(metadata))
        self.repo.commit({"engine.rs": "b\n"}, "engine change")
        with self.assertRaisesRegex(gamfit_version.VersionError, "between preparing the metadata and building"):
            gamfit_version.build_wheel(str(self.out("wheel")), None, str(metadata / name))

    def test_a_tree_that_moves_during_the_build_is_refused(self):
        os.chdir(self.repo.path)
        self.maturin.during_build = lambda: (self.repo.path / "engine.rs").write_text("edited\n")
        with self.assertRaisesRegex(gamfit_version.VersionError, "changed during the build"):
            gamfit_version.build_wheel(str(self.out("wheel")))


if __name__ == "__main__":
    unittest.main()

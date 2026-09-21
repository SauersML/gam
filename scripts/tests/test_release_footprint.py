"""Contracts for the release footprint guards, executed without building a release."""

import contextlib
import http.server
import importlib.util
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_footprint", ROOT / "scripts/release_footprint.py")
release_footprint = importlib.util.module_from_spec(SPEC)
sys.modules["release_footprint"] = release_footprint
SPEC.loader.exec_module(release_footprint)

GIB = release_footprint.GIB
MIB = release_footprint.MIB
LIMIT = release_footprint.PYPI_DEFAULT_PROJECT_LIMIT
# What gamfit's PyPI project held when this guard was written (69 releases),
# and one full release's worth of files (seven abi3 wheels, the cp314t wheels
# on the same seven platforms, the sdist).
PUBLISHED_TODAY = 8_289_947_778
FULL_RELEASE = {f"gamfit-0.2.0-{i}.whl": 32 * MIB for i in range(14)} | {"gamfit-0.2.0.tar.gz": 15 * MIB}


class DepInfoTests(unittest.TestCase):
    def test_every_prerequisite_is_read_including_escaped_spaces(self):
        text = (
            "/t/debug/deps/libgam_x-1.rmeta: crates/gam-x/src/lib.rs crates/gam-x/src/a\\ b.rs\n"
            "\n"
            "crates/gam-x/src/lib.rs:\n"
            "# env-dep:CARGO_PKG_VERSION=0.1\n"
        )
        self.assertEqual(
            release_footprint.parse_dep_info(text),
            {"crates/gam-x/src/lib.rs", "crates/gam-x/src/a b.rs"},
        )

    def test_paths_are_resolved_against_the_tree_and_outside_ones_dropped(self):
        with tempfile.TemporaryDirectory() as scratch:
            tree = pathlib.Path(scratch) / "gamfit-1.0.0"
            target = pathlib.Path(scratch) / "target" / "debug" / "deps"
            target.mkdir(parents=True)
            (target / "x.d").write_text(
                "out: crates/a/src/lib.rs crates/a/src/../../b/data.bin "
                f"{tree}/crates/c/src/lib.rs /home/.cargo/registry/src/faer/lib.rs\n"
            )
            self.assertEqual(
                release_footprint.read_by_compiler(target.parent.parent, tree),
                {"crates/a/src/lib.rs", "crates/b/data.bin", "crates/c/src/lib.rs"},
            )


class UnreadFileTests(unittest.TestCase):
    def test_manifests_and_package_metadata_need_not_be_compiled(self):
        for path in (
            "Cargo.toml", "Cargo.lock", "pyproject.toml", "PKG-INFO", "README_PYPI.md",
            "LICENSE", "rust-toolchain.toml", "crates/gam-x/Cargo.toml", "gamfit/__init__.py",
        ):
            self.assertTrue(release_footprint.is_packaging_metadata(path, "gamfit", set()), path)

    def test_the_in_tree_build_backend_is_read_by_pip_not_rustc(self):
        pyproject = {"build-system": {"build-backend": "gamfit_version", "backend-path": ["scripts"]}}
        backend = release_footprint.build_backend_files(pyproject)
        self.assertEqual(backend, {"scripts/gamfit_version.py", "scripts/gamfit_version/__init__.py"})
        self.assertTrue(release_footprint.is_packaging_metadata("scripts/gamfit_version.py", "gamfit", backend))
        self.assertFalse(release_footprint.is_packaging_metadata("scripts/other.py", "gamfit", backend))
        self.assertEqual(release_footprint.build_backend_files({}), set())

    def test_test_modules_fixtures_and_benches_are_refused(self):
        members = [
            "Cargo.toml",
            "crates/gam-x/src/lib.rs",
            "crates/gam-x/src/tests.rs",
            "crates/gam-x/tests/data/fixture.csv",
            "crates/gam-x/benches/fit.rs",
            "bench/run.py",
            "gamfit/__init__.py",
        ]
        self.assertEqual(
            release_footprint.unread_files(members, {"crates/gam-x/src/lib.rs"}, "gamfit", set()),
            [
                "bench/run.py",
                "crates/gam-x/benches/fit.rs",
                "crates/gam-x/src/tests.rs",
                "crates/gam-x/tests/data/fixture.csv",
            ],
        )

    def test_the_sdist_is_listed_relative_to_its_one_top_level_directory(self):
        with tempfile.TemporaryDirectory() as scratch:
            sdist = pathlib.Path(scratch) / "gamfit-1.0.0.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                for name in ("gamfit-1.0.0/Cargo.toml", "gamfit-1.0.0/crates/gam-x/src/tests.rs"):
                    info = tarfile.TarInfo(name)
                    archive.addfile(info, io.BytesIO(b""))
            self.assertEqual(
                release_footprint.sdist_members(sdist),
                ("gamfit-1.0.0", ["Cargo.toml", "crates/gam-x/src/tests.rs"]),
            )


class SdistExcludeTests(unittest.TestCase):
    """The pyproject rules the sdist check enforces on a real build."""

    def setUp(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            self.maturin = tomllib.load(handle)["tool"]["maturin"]

    def test_integration_tests_and_test_modules_are_excluded_from_the_sdist(self):
        excluded = {rule["path"] for rule in self.maturin["exclude"] if rule["format"] == "sdist"}
        for rule in ("crates/**/tests/**", "crates/**/tests.rs", "bench/**"):
            self.assertIn(rule, excluded)

    def test_every_kept_test_file_exists_and_is_opened_by_the_library(self):
        # A `!` rule for a file that no longer exists, or that no library
        # source names, would silently re-ship test code.
        for rule in self.maturin["exclude"]:
            if not rule["path"].startswith("!"):
                continue
            pattern = rule["path"][1:]
            matches = sorted(ROOT.glob(pattern))
            self.assertTrue(matches, pattern)
            for path in matches:
                src = path.parent
                named = path.stem if path.name != "test_support.rs" else "test_support"
                users = subprocess.run(
                    ["git", "grep", "-l", "-e", f'include!("{path.name}")', "-e", f"mod {named}", "--", str(src)],
                    cwd=ROOT, capture_output=True, text=True,
                ).stdout.split()
                self.assertTrue(users, f"{path.relative_to(ROOT)} is kept but nothing in {src.relative_to(ROOT)} uses it")

    def test_the_workspace_the_lockfile_describes_ships_whole(self):
        # Cargo.lock lists gam-cli, which the extension does not depend on;
        # without it `locked = true` refuses to build the published sdist.
        self.assertTrue(self.maturin["locked"])
        self.assertIn({"path": "crates/gam-cli/**/*", "format": "sdist"}, self.maturin["include"])
        with (ROOT / "Cargo.toml").open("rb") as handle:
            self.assertEqual(tomllib.load(handle)["workspace"]["members"], ["crates/*"])


class StorageVerdictTests(unittest.TestCase):
    def test_a_release_that_fits_under_the_threshold_proceeds(self):
        published = {"gamfit-1.0.0.tar.gz": PUBLISHED_TODAY}
        problems, usage = release_footprint.storage_verdict(published, FULL_RELEASE, LIMIT)
        self.assertEqual(problems, [])
        self.assertIn("of 10.00 GiB", usage)

    def test_a_release_past_the_threshold_is_refused(self):
        published = {"gamfit-1.0.0.tar.gz": int(9.5e9)}
        problems, _ = release_footprint.storage_verdict(published, FULL_RELEASE, LIMIT)
        self.assertEqual(len(problems), 1)
        self.assertIn("past the 90% release threshold", problems[0])

    def test_files_already_on_the_index_are_not_counted_twice(self):
        # The upload runs with --skip-existing, so a re-run adds nothing.
        published = {"gamfit-1.0.0.tar.gz": int(8.9e9), **FULL_RELEASE}
        problems, _ = release_footprint.storage_verdict(published, FULL_RELEASE, LIMIT)
        self.assertEqual(problems, [])

    def test_a_raised_limit_is_honoured(self):
        published = {"gamfit-1.0.0.tar.gz": int(9.5e9)}
        problems, _ = release_footprint.storage_verdict(published, FULL_RELEASE, 20 * GIB)
        self.assertEqual(problems, [])

    def test_a_file_over_the_per_file_limit_is_refused(self):
        problems, _ = release_footprint.storage_verdict({}, {"gamfit-0.2.0-big.whl": 101 * MIB}, LIMIT)
        self.assertEqual(len(problems), 1)
        self.assertIn("refuses files over 100 MiB", problems[0])


@contextlib.contextmanager
def simple_index(total_bytes):
    """A PEP 691 JSON simple index whose gamfit project holds ``total_bytes``."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/simple/gamfit/" or "application/vnd.pypi.simple.v1+json" not in self.headers["Accept"]:
                self.send_error(404)
                return
            body = json.dumps(
                {
                    "meta": {"api-version": "1.1"},
                    "name": "gamfit",
                    "files": [
                        {"filename": "gamfit-1.0.0.tar.gz", "size": total_bytes - total_bytes // 2, "yanked": True},
                        {"filename": "gamfit-1.0.1.tar.gz", "size": total_bytes // 2},
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.pypi.simple.v1+json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/simple/"
    finally:
        server.shutdown()
        server.server_close()


def release_guard_step() -> str:
    """The `run:` script of the release job's storage guard, as the runner gets it."""
    workflow = (ROOT / ".github/workflows/pypi-wheels.yml").read_text()
    match = re.search(
        r"- name: Refuse an upload that would fill the PyPI project\n(?:(?!      - ).*\n)*?        run: \|\n((?:          .*\n|\n)+)",
        workflow,
    )
    if match is None:
        raise AssertionError("pypi-wheels.yml has no storage guard step before the upload")
    return "\n".join(line[10:] for line in match.group(1).splitlines())


class ReleaseWorkflowDryRunTests(unittest.TestCase):
    """Run the release job's guard step as written, against a mocked index."""

    def dry_run(self, published_bytes, limit_gib=""):
        with tempfile.TemporaryDirectory() as scratch, simple_index(published_bytes) as index:
            work = pathlib.Path(scratch)
            (work / "release-src").symlink_to(ROOT)
            dist = work / "dist"
            dist.mkdir()
            # Sparse files: the guard reads sizes, never contents.
            for name, size in FULL_RELEASE.items():
                with (dist / name).open("wb") as handle:
                    handle.truncate(size)
            env = {
                **os.environ,
                "PYPI_INDEX_URL": index,
                "PYPI_PROJECT_SIZE_LIMIT_GIB": limit_gib,
                "PATH": f"{pathlib.Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            }
            return subprocess.run(
                ["bash", "-c", release_guard_step()], cwd=work, env=env, capture_output=True, text=True
            )

    def test_a_project_at_9_5_gb_stops_the_release_before_any_upload(self):
        result = self.dry_run(int(9.5e9))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("past the 90% release threshold", result.stderr)

    def test_the_project_as_it_stands_today_releases(self):
        result = self.dry_run(PUBLISHED_TODAY)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PyPI holds 7.72 GiB in 2 files", result.stdout)

    def test_a_limit_recorded_in_the_repository_variable_is_used(self):
        result = self.dry_run(int(9.5e9), limit_gib="20")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("of 20.00 GiB", result.stdout)


class WorkflowWiringTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / ".github/workflows/pypi-wheels.yml").read_text()

    def job(self, name):
        body = self.workflow[self.workflow.index(f"\n  {name}:\n") + 1 :]
        end = re.search(r"\n  [a-z]", body)
        return body[: end.start()] if end else body

    def test_the_guard_runs_after_the_metadata_check_and_before_the_upload(self):
        release = self.job("release")
        check = release.index("twine check --strict")
        guard = release.index("Refuse an upload that would fill the PyPI project")
        upload = release.index("-m twine upload")
        self.assertLess(check, guard)
        self.assertLess(guard, upload)
        self.assertIn("scripts/release_footprint.py", release[: release.index("download-artifact")])

    def test_the_published_sdist_is_footprint_checked_before_it_is_uploaded(self):
        sdist = self.job("sdist")
        self.assertLess(sdist.index("release_footprint.py sdist"), sdist.index("upload-artifact"))

    def test_every_pull_request_that_can_change_the_sdist_checks_it(self):
        workflow = (ROOT / ".github/workflows/release-footprint.yml").read_text()
        self.assertIn("release_footprint.py sdist", workflow)
        for path in ('"crates/**"', '"pyproject.toml"', '"Cargo.lock"', '"scripts/release_footprint.py"'):
            self.assertIn(path, workflow)


if __name__ == "__main__":
    unittest.main()

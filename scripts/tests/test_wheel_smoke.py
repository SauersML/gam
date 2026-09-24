"""Contracts for the installed-wheel smoke gate, executed without building a wheel."""

import importlib.util
import pathlib
import re
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("wheel_smoke", ROOT / "scripts/wheel_smoke.py")
wheel_smoke = importlib.util.module_from_spec(SPEC)
sys.modules["wheel_smoke"] = wheel_smoke
SPEC.loader.exec_module(wheel_smoke)

EXT = [".cpython-312-x86_64-linux-gnu.so", ".abi3.so", ".so"]
DIST_INFO = "gamfit-0.1.274.dist-info"


class PyprojectMatrixTests(unittest.TestCase):
    def setUp(self):
        self.project = wheel_smoke._pyproject()["project"]

    def test_advertised_versions_are_contiguous_from_the_requires_python_floor(self):
        versions = wheel_smoke.classifier_python_versions(self.project)
        floor = wheel_smoke.requires_python_floor(self.project)
        self.assertEqual(versions[0], floor)
        minors = [int(v.split(".")[1]) for v in versions]
        self.assertEqual(minors, list(range(minors[0], minors[-1] + 1)))

    def test_every_advertised_version_is_installed_on_the_reference_family(self):
        versions = wheel_smoke.classifier_python_versions(self.project)
        matrix = wheel_smoke.build_matrix(["linux-x86_64"], self.project)["include"]
        self.assertEqual([lane["python"] for lane in matrix if not lane["free_threaded"]], versions)

    def test_free_threaded_wheels_are_built_and_installed_on_every_family(self):
        self.assertEqual(wheel_smoke.free_threaded_python_versions(self.project), ["3.14t"])
        self.assertEqual(wheel_smoke.build_interpreters(self.project), ["3.10", "3.14t"])
        matrix = wheel_smoke.build_matrix(list(wheel_smoke.SCOPES["full"]), self.project)["include"]
        for family in wheel_smoke.FAMILIES:
            lanes = [lane for lane in matrix if lane["family"] == family and lane["free_threaded"]]
            self.assertEqual([lane["python"] for lane in lanes], ["3.14t"], family)
            self.assertEqual({lane["resolution"] for lane in lanes}, {"highest"}, family)

    def test_free_threaded_lanes_follow_the_classifier(self):
        classifiers = [c for c in self.project["classifiers"] if "Free Threading" not in c]
        project = dict(self.project, classifiers=classifiers)
        self.assertEqual(wheel_smoke.free_threaded_python_versions(project), [])
        self.assertEqual(wheel_smoke.build_interpreters(project), ["3.10"])
        matrix = wheel_smoke.build_matrix(["linux-x86_64"], project)["include"]
        self.assertFalse(any(lane["free_threaded"] for lane in matrix))

    def test_the_workflows_read_the_matrix_and_interpreters_from_the_command_line(self):
        # The workflows run the CLI, not build_matrix: it must read the
        # [project] table itself.
        script = ROOT / "scripts/wheel_smoke.py"
        for args in (["matrix", "--scope", "full"], ["matrix", "--families", "linux-x86_64"], ["interpreters"]):
            done = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), wheel_smoke.build_interpreters(self.project))

    def test_every_family_brackets_the_range_with_the_floor_on_lowest_requirements(self):
        versions = wheel_smoke.classifier_python_versions(self.project)
        matrix = wheel_smoke.build_matrix(list(wheel_smoke.SCOPES["full"]), self.project)["include"]
        for family in wheel_smoke.FAMILIES:
            lanes = {lane["python"]: lane["resolution"] for lane in matrix if lane["family"] == family}
            self.assertEqual(lanes[versions[0]], "lowest", family)
            self.assertEqual(lanes[versions[-1]], "highest", family)

    def test_floor_that_disagrees_with_requires_python_is_refused(self):
        project = dict(self.project, **{"requires-python": ">=3.9"})
        with self.assertRaisesRegex(ValueError, "disagrees with requires-python"):
            wheel_smoke.build_matrix(["linux-x86_64"], project)

    def test_scopes_name_only_known_families(self):
        for scope, families in wheel_smoke.SCOPES.items():
            self.assertTrue(set(families) <= set(wheel_smoke.FAMILIES), scope)
        self.assertEqual(set(wheel_smoke.SCOPES["full"]), set(wheel_smoke.FAMILIES))

    def test_every_family_is_an_artifact_pypi_wheels_uploads(self):
        workflow = (ROOT / ".github/workflows/pypi-wheels.yml").read_text()
        uploaded = set()
        for prefix, targets in (
            ("linux", ("x86_64", "aarch64")),
            ("musllinux", ("x86_64", "aarch64")),
            ("macos", ("x86_64", "aarch64")),
            ("windows", ("x64",)),
        ):
            self.assertRegex(workflow, rf"name: wheels-{prefix}-\$\{{\{{ matrix\.")
            uploaded |= {f"{prefix}-{target}" for target in targets}
        self.assertEqual(uploaded, set(wheel_smoke.FAMILIES))


class RequirementTests(unittest.TestCase):
    def test_extras_are_not_runtime_requirements(self):
        requires = [
            "numpy>=1.26",
            'pandas>=2.0 ; extra == "pandas"',
            "nvidia-cublas-cu12<13,>=12.0 ; (platform_system == 'Linux' and platform_machine == 'x86_64') and extra == 'cuda'",
        ]
        self.assertEqual(wheel_smoke.runtime_requirements(requires), {"numpy"})

    def test_unconditional_marker_is_refused_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "environment marker"):
            wheel_smoke.runtime_requirements(["tomli>=2 ; python_version < '3.11'"])

    def test_closure_follows_transitive_runtime_requirements(self):
        graph = {"gamfit": ["numpy>=1.26", "Foo_Bar"], "numpy": None, "foo-bar": ["numpy"]}
        closure = wheel_smoke.requirement_closure("gamfit", graph.__getitem__)
        self.assertEqual(closure, {"gamfit", "numpy", "foo-bar"})


class WheelSelectionTests(unittest.TestCase):
    WHEELS = [
        pathlib.Path("gamfit-0.1.274-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"),
        pathlib.Path("gamfit-0.1.274-cp313-cp313t-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"),
        pathlib.Path("gamfit-0.1.274-cp314-cp314t-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"),
    ]

    def test_gil_interpreters_install_the_abi3_wheel(self):
        for tag in ("cp310", "cp313", "cp314"):
            self.assertEqual(wheel_smoke.wheel_for(self.WHEELS, tag, False), self.WHEELS[0], tag)

    def test_free_threaded_interpreters_install_their_own_wheel(self):
        self.assertEqual(wheel_smoke.wheel_for(self.WHEELS, "cp313", True), self.WHEELS[1])
        self.assertEqual(wheel_smoke.wheel_for(self.WHEELS, "cp314", True), self.WHEELS[2])

    def test_a_missing_or_ambiguous_wheel_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "exactly one cp315t"):
            wheel_smoke.wheel_for(self.WHEELS, "cp315", True)
        with self.assertRaisesRegex(SystemExit, "exactly one abi3"):
            wheel_smoke.wheel_for(self.WHEELS[:1] * 2, "cp310", False)


class WheelContentTests(unittest.TestCase):
    def test_package_sources_marker_and_extension_are_clean(self):
        paths = [
            "gamfit/__init__.py",
            "gamfit/torch/fit.py",
            "gamfit/_rust.pyi",
            "gamfit/py.typed",
            "gamfit/_rust.abi3.so",
            "gamfit/__pycache__/_api.cpython-312.pyc",
            f"{DIST_INFO}/METADATA",
            f"{DIST_INFO}/licenses/LICENSE",
            f"{DIST_INFO}/sboms/gam-pyffi.cyclonedx.json",
        ]
        self.assertEqual(wheel_smoke.stray_wheel_files(paths, DIST_INFO, EXT), [])

    def test_auditwheel_vendored_libraries_are_part_of_a_repaired_wheel(self):
        vendored = ["gamfit.libs/libgcc_s-f685abf1.so.1", "gamfit.libs/libgomp-a34b3233.so.1.0.0"]
        self.assertEqual(wheel_smoke.stray_wheel_files(vendored, DIST_INFO, EXT), [])
        not_libraries = ["gamfit.libs/NOTES.txt", "gamfit.libs/nested/libz.so.1"]
        self.assertEqual(wheel_smoke.stray_wheel_files(not_libraries, DIST_INFO, EXT), not_libraries)

    def test_stray_files_are_reported(self):
        stray = [
            "tests/test_fit.py",
            "gamfit/tests/data/fixture.csv",
            "gamfit/src/lib.rs",
            "gamfit/examples/demo.ipynb",
            "gamfit/torch/_rust.abi3.so",
            "gamfit/_other.abi3.so",
            "README.md",
        ]
        self.assertEqual(wheel_smoke.stray_wheel_files(stray, DIST_INFO, EXT), stray)


class WorkflowWiringTests(unittest.TestCase):
    def test_pypi_upload_waits_for_the_installed_wheel_smoke_test(self):
        workflow = (ROOT / ".github/workflows/pypi-wheels.yml").read_text()
        release = workflow[workflow.index("\n  release:") :]
        needs = release[: release.index("runs-on:")]
        self.assertIn("- smoke", needs)
        self.assertIn("needs.smoke.result == 'success'", needs)
        self.assertRegex(release, r"twine check --strict dist/\*")

    def test_every_pypi_builder_builds_the_free_threaded_interpreters(self):
        workflow = (ROOT / ".github/workflows/pypi-wheels.yml").read_text()
        for job in ("linux", "musllinux", "macos", "windows"):
            body = workflow[workflow.index(f"\n  {job}:\n") :]
            body = body[: body.index("upload-artifact")]
            self.assertRegex(body, r"-i \$\{\{ (needs\.gate|steps\.interpreters)\.outputs\.", job)
        self.assertTrue("python3 scripts/wheel_smoke.py interpreters" in workflow, "pypi-wheels.yml")

    def test_the_rolling_build_ships_and_smokes_free_threaded_wheels(self):
        workflow = (ROOT / ".github/workflows/wheel-nightly.yml").read_text()
        for needle in ("python3 scripts/wheel_smoke.py interpreters", "gamfit-free-threaded-"):
            self.assertTrue(needle in workflow, f"wheel-nightly.yml lacks {needle!r}")

    def test_smoke_jobs_run_the_shared_script_without_the_package_checkout(self):
        for name in ("pypi-wheels.yml", "wheel-nightly.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text()
            smoke = workflow[workflow.index("\n  smoke:") :]
            self.assertIn("sparse-checkout: |\n            scripts/wheel_smoke.py", smoke, name)
            self.assertRegex(smoke, r"wheel_smoke\.py run ", name)
            self.assertIsNone(re.search(r"pip install[^\n]*\.\[", smoke), name)


if __name__ == "__main__":
    unittest.main()

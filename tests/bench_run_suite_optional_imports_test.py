import builtins
import importlib.util
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_SUITE_PATH = _REPO_ROOT / "bench" / "run_suite.py"


class RunSuiteOptionalImportTests(unittest.TestCase):
    def test_import_does_not_require_optional_survival_packages(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "bench_run_suite_optional_import_guard",
            _RUN_SUITE_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load benchmark runner from {_RUN_SUITE_PATH}")
        module = importlib.util.module_from_spec(spec)
        orig_import = builtins.__import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".", 1)[0]
            if root in {"lifelines", "sklearn"}:
                raise ModuleNotFoundError(f"No module named '{root}'")
            return orig_import(name, globals, locals, fromlist, level)

        try:
            builtins.__import__ = _guarded_import
            spec.loader.exec_module(module)
        finally:
            builtins.__import__ = orig_import
            sys.modules.pop(spec.name, None)

        self.assertTrue(callable(module.main))

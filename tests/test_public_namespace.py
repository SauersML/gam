"""The top-level ``gamfit`` namespace is a deliberate, pinned API (PKG-06).

``import gamfit`` exposes only the fit / load entry points and the fitted-model
classes. Everything else lives in a public submodule loaded on first access, and
every other module in the package is private (underscore-prefixed).
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit

TOP_LEVEL = {
    "CtnStage1",
    "EventHistoryModel",
    "JointEventModel",
    "Model",
    "MultinomialModel",
    "ResponseGeometryModel",
    "__version__",
    "build_info",
    "compare_models",
    "competing_risks_cif",
    "explain_error",
    "fit",
    "fit_array",
    "fit_event_history",
    "fit_joint_event_model",
    "load",
    "load_joint_event_model",
    "loads",
    "validate_formula",
}

SUBMODULES = {
    "basis",
    "cuda",
    "diagnostics",
    "errors",
    "examples",
    "geometry",
    "identifiability",
    "inference",
    "kernels",
    "kernels_jax",
    "kernels_torch",
    "manifolds",
    "penalties",
    "plot",
    "reml",
    "response_geometry",
    "results",
    "sae",
    "sklearn",
    "smooth",
    "topology",
    "torch",
}

PACKAGE_DIR = pathlib.Path(gamfit.__file__).parent


def test_top_level_all_is_pinned() -> None:
    assert set(gamfit.__all__) == TOP_LEVEL
    assert len(gamfit.__all__) == len(TOP_LEVEL)


def test_public_submodules_are_exactly_the_non_underscore_modules() -> None:
    on_disk = {
        path.stem if path.is_file() else path.name
        for path in PACKAGE_DIR.iterdir()
        if not path.name.startswith("_")
        and (path.suffix == ".py" or (path / "__init__.py").is_file())
    }
    assert on_disk == SUBMODULES
    assert gamfit._SUBMODULES == SUBMODULES


def test_dir_lists_the_api_and_the_submodules() -> None:
    assert set(dir(gamfit)) == TOP_LEVEL | SUBMODULES


@pytest.mark.parametrize(
    "name",
    [
        # Duplicates or internals that used to leak to the top level.
        "save",
        "identifiability_check",
        "TopologySphere",
        "SmoothSpec",
        "plot_atom",
        "plot_fit",
        "emit_inference_warnings",
        "SUPPORT_SAE_SCHEMA",
        "PoincareAtoms",
        "InterchangeSwapDecoder",
        # Names that now live only in their submodule.
        "GamError",
        "Diagnostics",
        "PosteriorSamples",
        "BSpline",
        "sae_manifold_fit",
        "select_topology",
        "gaussian_reml_fit",
        "bspline_basis",
    ],
)
def test_removed_names_are_not_top_level(name: str) -> None:
    assert not hasattr(gamfit, name)


def test_each_name_has_one_public_home() -> None:
    assert gamfit.errors.GamError.__name__ == "GamError"
    assert gamfit.topology.Sphere is not gamfit.smooth.Sphere
    for sub in SUBMODULES - {"torch", "sklearn", "kernels_jax", "kernels_torch"}:
        module = getattr(gamfit, sub)
        exported = getattr(module, "__all__", [])
        missing = [name for name in exported if not hasattr(module, name)]
        assert not missing, f"gamfit.{sub}.__all__ names missing attributes: {missing}"


def test_import_gamfit_does_not_load_research_modules_or_matplotlib() -> None:
    code = (
        "import sys, gamfit\n"
        "heavy = ['gamfit._sae_spectral', 'gamfit._sparse_dictionary',"
        " 'gamfit._select_topology', 'gamfit.sae', 'gamfit.plot', 'matplotlib', 'torch']\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_plot_is_a_submodule_that_imports_without_matplotlib(monkeypatch: typing.Any) -> None:
    for name in [m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.delitem(sys.modules, "gamfit.plot", raising=False)
    monkeypatch.delattr(gamfit, "plot", raising=False)

    plot = importlib.import_module("gamfit.plot")
    assert set(plot.__all__) == {"model", "trace", "sae_atom", "sae_fit"}

    model = gamfit.fit({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.1, 0.9, 2.1, 2.9, 4.2, 5.0]}, "y ~ x")
    with pytest.raises(ImportError, match=r"pip install 'gamfit\[plot\]'"):
        plot.model(model, {"x": [0.0, 1.0], "y": [0.0, 1.0]})
    with pytest.raises(ImportError, match=r"pip install 'gamfit\[plot\]'"):
        model.plot({"x": [0.0, 1.0], "y": [0.0, 1.0]})

import contextlib
import typing
import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_SUITE_PATH = _REPO_ROOT / "bench" / "run_suite.py"
_SPEC = importlib.util.spec_from_file_location("bench_run_suite", _RUN_SUITE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load benchmark runner from {_RUN_SUITE_PATH}")
_RUN_SUITE: typing.Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_SUITE)
_GITIGNORE_PATH = _REPO_ROOT / ".gitignore"
_BENCH_DATASET_DIR = _REPO_ROOT / "bench" / "datasets"
_REQUIRED_BENCHMARK_DATASETS = {
    "31_day.csv",
    "bone.csv",
    "cirrhosis.csv",
    "five_day.csv",
    "haberman.csv",
    "heart_failure_clinical_records_dataset.csv",
    "horse.csv",
    "icu_survival_death.csv",
    "icu_survival_los.csv",
    "lidar.csv",
    "prostate.csv",
    "wine.csv",
}


@contextlib.contextmanager
def _patched_attrs(*patches: tuple[typing.Any, str, typing.Any]) -> typing.Iterator[None]:
    originals = [(obj, name, getattr(obj, name)) for obj, name, _value in patches]
    try:
        for obj, name, value in patches:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(originals):
            setattr(obj, name, value)


def _write_scenarios(workdir: Path, scenarios: list[dict[str, typing.Any]]) -> tuple[Path, Path]:
    scenario_path = workdir / "scenarios.json"
    out_path = workdir / "results.json"
    scenario_path.write_text(json.dumps({"scenarios": scenarios}))
    return scenario_path, out_path


def _parse_args(scenario_path: Path, out_path: Path) -> typing.Any:
    return lambda _self: SimpleNamespace(
        scenarios=scenario_path,
        out=out_path,
        scenario_names=None,
    )


def _tempdir_factory(root: Path) -> typing.Any:
    return lambda prefix="": tempfile.TemporaryDirectory(prefix=prefix, dir=root)


def _main_patches(
    scenario_path: Path,
    out_path: Path,
    dataset: dict[str, typing.Any],
    enabled: typing.Callable[[dict[str, typing.Any], str], bool],
    *,
    datapoint_figures: bool = False,
) -> list[tuple[typing.Any, str, typing.Any]]:
    patches = [
        (_RUN_SUITE.argparse.ArgumentParser, "parse_args", _parse_args(scenario_path, out_path)),
        (_RUN_SUITE, "dataset_for_scenario", lambda _scenario: dataset),
        (_RUN_SUITE, "folds_for_dataset", lambda _ds: []),
        (_RUN_SUITE, "_assert_basis_parity_for_scenario", lambda *args, **kwargs: None),
        (_RUN_SUITE, "build_shared_fold_artifacts", lambda *args, **kwargs: []),
        (_RUN_SUITE, "_is_contender_enabled", enabled),
        (_RUN_SUITE, "generate_scenario_figures", lambda *_args, **_kwargs: []),
        (_RUN_SUITE, "zip_figure_dir", lambda *_args, **_kwargs: None),
    ]
    if datapoint_figures:
        patches.append((_RUN_SUITE, "generate_scenario_datapoint_figures", lambda *_args, **_kwargs: []))
    return patches


class RunSuiteMappingTests(unittest.TestCase):
    def test_terminal_output_sanitizer_removes_cursor_controls_across_chunks(self) -> None:
        sanitizer = _RUN_SUITE._TerminalOutputSanitizer()
        text = (
            sanitizer.feed("progress\r        [1s] ok \x1b[")
            + sanitizer.feed("2K next\x1b]0;title")
            + sanitizer.feed("\x07 done\n")
        )
        self.assertEqual(text, "progress\n[1s] ok  next done\n")

    def test_finalize_cv_result_keeps_evaluation_from_fold_count(self) -> None:
        result = _RUN_SUITE._finalize_cv_result(
            contender="rust_gam",
            scenario_name="wine_temp_vs_year",
            family="gaussian",
            cv_rows=[
                {
                    "fit_sec": 0.1,
                    "predict_sec": 0.01,
                    "logloss": 1.0,
                    "mse": 0.25,
                    "rmse": 0.5,
                    "mae": 0.4,
                    "r2": 0.2,
                    "n_test": 10,
                }
                for _ in range(5)
            ],
            plot_payload=None,
            model_spec="s_temp ~ s(year, type=ps, knots=7) via release binary [5-fold CV]",
        )
        self.assertEqual(result["evaluation"], "5-fold CV")
        # Required-metadata contract: every successful result row must carry
        # the metadata fields downstream tooling (validate_result_metadata,
        # the JSON aggregator, the report generator) depends on. A mutation
        # that drops any of these would silently corrupt aggregated reports.
        for required in ("status", "scenario_name", "contender", "evaluation", "model_spec"):
            self.assertIn(
                required,
                result,
                f"finalized CV result missing required key {required!r}; got {sorted(result)}",
            )
        self.assertEqual(result["scenario_name"], "wine_temp_vs_year")
        self.assertEqual(result["contender"], "rust_gam")

    def test_finalize_cv_result_rejects_reserved_metric_keys(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reserved result keys: evaluation"):
            _RUN_SUITE._finalize_cv_result(
                contender="rust_gam",
                scenario_name="wine_temp_vs_year",
                family="gaussian",
                cv_rows=[
                    {
                        "fit_sec": 0.1,
                        "predict_sec": 0.01,
                        "logloss": 1.0,
                        "mse": 0.25,
                        "rmse": 0.5,
                        "mae": 0.4,
                        "r2": 0.2,
                        "n_test": 10,
                    }
                ],
                plot_payload=None,
                model_spec="s_temp ~ s(year, type=ps, knots=7) via release binary [holdout]",
                extra_metrics={"evaluation": "broken"},
            )

    def test_validate_result_metadata_accepts_matching_cv_spec(self) -> None:
        _RUN_SUITE._validate_result_metadata(
            [
                {
                    "status": "ok",
                    "contender": "rust_gam",
                    "scenario_name": "lidar_semipar",
                    "evaluation": "5-fold CV",
                    "model_spec": "logratio ~ s(range, type=ps, knots=24) via release binary [5-fold CV]",
                }
            ]
        )

    def test_validate_result_metadata_rejects_missing_evaluation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "model result metadata/spec mismatch for rust_gam / lidar_semipar"):
            _RUN_SUITE._validate_result_metadata(
                [
                    {
                        "status": "ok",
                        "contender": "rust_gam",
                        "scenario_name": "lidar_semipar",
                        "evaluation": None,
                        "model_spec": "logratio ~ s(range, type=ps, knots=24) via release binary [5-fold CV]",
                    }
                ]
            )

    def test_flexible_gamlss_failures_are_non_blocking(self) -> None:
        self.assertTrue(
            _RUN_SUITE._is_non_blocking_failure(
                {
                    "status": "error",
                    "scenario_name": "bone_gamair",
                    "contender": "rust_gamlss_flexible",
                    "error": "predict_posterior_mean failed",
                }
            )
        )

    def test_r_gamlss_sigma_formula_rejects_constant_sigma(self) -> None:
        ds = {
            "rows": [{"y": 1.0}],
            "features": [],
            "target": "y",
            "family": "gaussian",
        }
        with self.assertRaisesRegex(RuntimeError, "requires a non-constant sigma model"):
            _RUN_SUITE._sigma_feature_formula(ds, scenario_name="toy", backend="r_gamlss")

    def test_run_external_r_gamlss_cv_supports_binomial_family(self) -> None:
        scenario = {"name": "small_dense"}
        ds = {
            "family": "binomial",
            "rows": [
                {"x1": 0.0, "x2": 0.0, "y": 0.0},
                {"x1": 1.0, "x2": 1.0, "y": 1.0},
            ],
            "features": ["x1", "x2"],
            "target": "y",
        }
        folds = [SimpleNamespace(train_idx=[0], test_idx=[1])]
        seen_scripts: list[str] = []

        def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any = None) -> typing.Any:
            if cmd[0] == "Rscript":
                seen_scripts.append(Path(cmd[1]).read_text())
                Path(cmd[3]).write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "fit_sec": 0.1,
                            "predict_sec": 0.01,
                            "pred": [0.8],
                            "sigma": [1.2],
                            "model_spec": "gamlss(BI; sigma.formula=~ pb(x1) + pb(x2)): y ~ pb(x1)",
                        }
                    )
                )
                return 0, "", ""
            return 1, "", f"unexpected command: {cmd}"

        with tempfile.TemporaryDirectory() as td:
            with _patched_attrs(
                (_RUN_SUITE, "run_cmd", _fake_run_cmd),
                (_RUN_SUITE, "_workspace_tempdir", _tempdir_factory(Path(td))),
            ):
                result = _RUN_SUITE.run_external_r_gamlss_cv(scenario, ds=ds, folds=folds)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["family"], "binomial")
        self.assertTrue(any("family = fit_family" in script for script in seen_scripts))
        self.assertTrue(any("BI()" in script for script in seen_scripts))

    def test_validate_scenario_schema_accepts_every_canonical_scenario(self) -> None:
        # Pure-Python schema dispatch must accept every scenario in the
        # committed bench/scenarios.json. CI runs this preflight in the
        # benchmark-prepare job, which does not build the Rust extension and
        # does not download the CSV fixtures — so the validator must never
        # touch the data-generation path. A regression where a new scenario
        # name pattern lands in scenarios.json without a matching dispatch
        # branch in `_resolve_scenario_loader` will fail here, but the
        # validator itself must NOT raise `gamfit Rust extension is not built`
        # for any registered name.
        scenarios_path = _REPO_ROOT / "bench" / "scenarios.json"
        cfg = json.loads(scenarios_path.read_text())
        scenarios = cfg.get("scenarios", [])
        self.assertGreater(len(scenarios), 0)
        for s in scenarios:
            _RUN_SUITE.validate_scenario_schema(s)

    def test_validate_scenario_schema_rejects_unknown_name(self) -> None:
        with self.assertRaises(RuntimeError):
            _RUN_SUITE.validate_scenario_schema({"name": "does_not_exist_xyz"})

    def test_validate_scenario_schema_rejects_missing_name(self) -> None:
        with self.assertRaises(RuntimeError):
            _RUN_SUITE.validate_scenario_schema({})

    def test_validate_scenario_schema_rejects_non_dict(self) -> None:
        with self.assertRaises(RuntimeError):
            _RUN_SUITE.validate_scenario_schema("not a dict")  # type: ignore[arg-type]

    def test_required_benchmark_datasets_exist(self) -> None:
        missing = sorted(
            dataset_name
            for dataset_name in _REQUIRED_BENCHMARK_DATASETS
            if not (_BENCH_DATASET_DIR / dataset_name).exists()
        )
        self.assertEqual(missing, [])

    def test_checked_in_benchmark_datasets_are_not_gitignored(self) -> None:
        ignored_entries = {
            line.strip()
            for line in _GITIGNORE_PATH.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        ignored_required = sorted(
            f"bench/datasets/{dataset_name}"
            for dataset_name in _REQUIRED_BENCHMARK_DATASETS
            if f"bench/datasets/{dataset_name}" in ignored_entries
        )
        self.assertEqual(ignored_required, [])

    def assert_joint_mapping(self, scenario_name: str, expected_dim: int, expected_knots: int) -> None:
        cfg = _RUN_SUITE._scenario_fit_mapping(scenario_name)
        self.assertIsNotNone(cfg, scenario_name)
        self.assertEqual(cfg["smooth_basis"], "thinplate")
        self.assertEqual(len(cfg["smooth_cols"]), expected_dim)
        self.assertEqual(cfg["knots"], expected_knots)

    def test_geo_disease_eas_tp_uses_full_joint_embedding(self) -> None:
        # Joint-PC contract: TPS-over-PCs uses the full PC count uniformly
        # (16 for `eas`, 3 for `eas3`). Canonical TPS at d=16 with k≤24 is
        # mathematically infeasible, but the Rust basis builder auto-
        # promotes such requests to a pure Duchon spline (Riesz-fractional
        # generalization) with finite kernel diagonal — see
        # `duchon_thin_plate_fallback_params` in src/terms/basis.rs.
        self.assert_joint_mapping("geo_disease_eas_tp_k6", expected_dim=16, expected_knots=6)
        self.assert_joint_mapping("geo_disease_eas_tp_k12", expected_dim=16, expected_knots=12)
        self.assert_joint_mapping("geo_disease_eas_tp_k24", expected_dim=16, expected_knots=24)
        self.assert_joint_mapping("geo_disease_eas3_tp_k6", expected_dim=3, expected_knots=6)
        self.assert_joint_mapping("geo_disease_eas3_tp_k24", expected_dim=3, expected_knots=24)

    def test_geo_latlon_tp_uses_full_joint_embedding(self) -> None:
        self.assert_joint_mapping("geo_latlon_equatornoise_tp_k12", expected_dim=6, expected_knots=12)
        self.assert_joint_mapping("geo_latlon_superpopnoise_tp_k24", expected_dim=6, expected_knots=24)

    def test_papuan_and_subpop_tp_use_full_joint_embedding(self) -> None:
        self.assert_joint_mapping("papuan_oce4_tp_k12", expected_dim=4, expected_knots=12)
        self.assert_joint_mapping("geo_subpop16_tp_k24", expected_dim=16, expected_knots=24)

    def test_geo_subpop16_dataset_builds_without_external_pc_file(self) -> None:
        ds = _RUN_SUITE.dataset_for_scenario({"name": "geo_subpop16_tp_k6"})
        self.assertEqual(ds["family"], "binomial")
        self.assertEqual(ds["features"], [f"pc{i}" for i in range(1, 17)])
        self.assertGreater(len(ds["rows"]), 0)

    def test_geo_latlon_dataset_builds_without_external_pc_file(self) -> None:
        ds = _RUN_SUITE.dataset_for_scenario({"name": "geo_latlon_superpopnoise_tp_k12"})
        self.assertEqual(ds["family"], "binomial")
        self.assertEqual(ds["features"], [f"pc{i}" for i in range(1, 7)])
        self.assertGreater(len(ds["rows"]), 0)

    def test_geo_subpop16_marginal_slope_aniso_keeps_16d_duchon_mapping(self) -> None:
        cfg = _RUN_SUITE._scenario_fit_mapping("geo_subpop16_margslope_aniso_duchon16d_k50")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["family"], "binomial-logit")
        self.assertEqual(cfg["smooth_basis"], "duchon")
        self.assertEqual(cfg["smooth_cols"], [f"pc{i}" for i in range(1, 17)])
        self.assertEqual(cfg["knots"], 50)
        self.assertTrue(cfg.get("scale_dimensions"))
        term = _RUN_SUITE._rust_joint_spatial_term(
            cfg["smooth_basis"],
            cfg["smooth_cols"],
            cfg["knots"],
            ", double_penalty=true",
        )
        self.assertIn("order=0", term)
        self.assertIn("power=8", term)
        self.assertIn("length_scale=1.0", term)
        self.assertNotIn("double_penalty", term)

    def test_geo_subpop16_marginal_slope_aniso_lane_is_present_and_enabled(self) -> None:
        scenarios = json.loads((_REPO_ROOT / "bench" / "scenarios.json").read_text())["scenarios"]
        scenario = next(
            s for s in scenarios if s["name"] == "geo_subpop16_margslope_aniso_duchon16d_k50"
        )
        self.assertTrue(
            _RUN_SUITE._is_contender_enabled(scenario, "rust_gamlss_marginal_slope")
        )

    def test_thread3_adaptive_reml_uses_current_boolean_cli(self) -> None:
        seen = []

        def _fake_run_rust_scenario_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            if kwargs.get("contender_name") == "rust_thread3_adaptive_reml":
                seen.append(list(kwargs.get("rust_fit_extra_args") or []))
            return {
                "status": "ok",
                "scenario_name": "thread3_admixture_cliff",
                "contender": kwargs.get("contender_name", "rust_gam"),
                "model_spec": "5-fold CV",
                "evaluation": "cv",
            }

        with tempfile.TemporaryDirectory() as td:
            scenario_path, out_path = _write_scenarios(Path(td), [{"name": "thread3_admixture_cliff"}])
            with _patched_attrs(
                (_RUN_SUITE, "run_rust_scenario_cv", _fake_run_rust_scenario_cv),
                (_RUN_SUITE.argparse.ArgumentParser, "parse_args", _parse_args(scenario_path, out_path)),
                (
                    _RUN_SUITE,
                    "dataset_for_scenario",
                    lambda _scenario: {
                        "rows": [{"y": 0.0}],
                        "features": ["pc1"],
                        "target": "y",
                        "family": "binomial",
                    },
                ),
                (_RUN_SUITE, "folds_for_dataset", lambda _ds: []),
                (_RUN_SUITE, "_assert_basis_parity_for_scenario", lambda *args, **kwargs: None),
                (_RUN_SUITE, "build_shared_fold_artifacts", lambda *args, **kwargs: []),
                (
                    _RUN_SUITE,
                    "_is_contender_enabled",
                    lambda _scenario, contender: contender == "rust_thread3_adaptive_reml",
                ),
                (_RUN_SUITE, "generate_scenario_figures", lambda *_args, **_kwargs: []),
                (_RUN_SUITE, "zip_figure_dir", lambda *_args, **_kwargs: None),
            ):
                _RUN_SUITE.main()

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], ["--adaptive-regularization", "true"])

    def test_thread3_adaptive_flexible_passes_flexible_formula_link(self) -> None:
        seen_formula_links = []

        def _fake_run_rust_scenario_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            if kwargs.get("contender_name") == "rust_thread3_adaptive_reml_flexible":
                seen_formula_links.append(kwargs.get("formula_link"))
            return {
                "status": "ok",
                "scenario_name": "thread3_admixture_cliff",
                "contender": kwargs.get("contender_name", "rust_gam"),
                "model_spec": "5-fold CV",
                "evaluation": "5-fold CV",
            }

        with tempfile.TemporaryDirectory() as td:
            scenario_path, out_path = _write_scenarios(Path(td), [{"name": "thread3_admixture_cliff"}])
            with _patched_attrs(
                (_RUN_SUITE, "run_rust_scenario_cv", _fake_run_rust_scenario_cv),
                (_RUN_SUITE.argparse.ArgumentParser, "parse_args", _parse_args(scenario_path, out_path)),
                (
                    _RUN_SUITE,
                    "dataset_for_scenario",
                    lambda _scenario: {
                        "rows": [{"y": 0.0, "pc1": 0.0}],
                        "features": ["pc1"],
                        "target": "y",
                        "family": "binomial",
                    },
                ),
                (_RUN_SUITE, "folds_for_dataset", lambda _ds: []),
                (_RUN_SUITE, "_assert_basis_parity_for_scenario", lambda *args, **kwargs: None),
                (_RUN_SUITE, "build_shared_fold_artifacts", lambda *args, **kwargs: []),
                (
                    _RUN_SUITE,
                    "_is_contender_enabled",
                    lambda _scenario, contender: contender
                    in {"rust_gam", "rust_thread3_adaptive_reml_flexible"},
                ),
                (_RUN_SUITE, "generate_scenario_figures", lambda *_args, **_kwargs: []),
                (_RUN_SUITE, "generate_scenario_datapoint_figures", lambda *_args, **_kwargs: []),
                (_RUN_SUITE, "zip_figure_dir", lambda *_args, **_kwargs: None),
            ):
                _RUN_SUITE.main()

        self.assertEqual(
            seen_formula_links,
            [
                _RUN_SUITE._flexible_link_name(
                    _RUN_SUITE._default_rust_formula_link_for_family("binomial")
                )
            ],
        )

    def test_binomial_formula_link_defaults_to_probit(self) -> None:
        self.assertEqual(
            _RUN_SUITE._default_rust_formula_link_for_family("binomial"), "probit"
        )

    def test_survival_benchmark_fit_options_require_structural_ispline_basis(self) -> None:
        expected = {
            "icu_survival_death": 10,
            "icu_survival_los": 10,
            "heart_failure_survival": 8,
            "cirrhosis_survival": 8,
        }
        for scenario_name, expected_knots in expected.items():
            with self.subTest(scenario_name=scenario_name):
                cfg = _RUN_SUITE._rust_survival_fit_options_for_scenario(scenario_name)
                self.assertEqual(cfg["time_basis"], "ispline")
                self.assertEqual(cfg["time_degree"], 3)
                self.assertEqual(cfg["time_num_internal_knots"], expected_knots)
                self.assertGreaterEqual(cfg["time_smooth_lambda"], 0.0)

    def test_survival_benchmark_cli_args_emit_ispline(self) -> None:
        args = _RUN_SUITE._rust_survival_fit_cli_args("icu_survival_death")
        self.assertIn("--time-basis", args)
        idx = args.index("--time-basis")
        self.assertEqual(args[idx + 1], "ispline")

    def test_run_rust_scenario_cv_rejects_survival_misuse(self) -> None:
        scenario = {"name": "heart_failure_survival"}
        ds = _RUN_SUITE.dataset_for_scenario(scenario)
        with self.assertRaisesRegex(RuntimeError, "run_rust_gamlss_survival_cv"):
            _RUN_SUITE.run_rust_scenario_cv(scenario, ds=ds, folds=[])

    def test_run_rust_scenario_cv_emits_evaluation_metadata(self) -> None:
        scenario = {"name": "lidar_semipar"}
        ds = {
            "family": "gaussian",
            "rows": [
                {"range": 0.0, "logratio": 0.0},
                {"range": 1.0, "logratio": 1.0},
                {"range": 2.0, "logratio": 2.0},
                {"range": 3.0, "logratio": 3.0},
            ],
            "features": ["range"],
            "target": "logratio",
        }
        folds = [
            SimpleNamespace(train_idx=[0, 1], test_idx=[2, 3]),
            SimpleNamespace(train_idx=[2, 3], test_idx=[0, 1]),
        ]
        def _fake_run_cmd(cmd: typing.Any, cwd: typing.Any = None) -> typing.Any:
            if len(cmd) >= 2 and cmd[1] == "fit":
                Path(cmd[cmd.index("--out") + 1]).write_text(
                    json.dumps({"fit_result": {"standard_deviation": 1.25}})
                )
                return 0, "", ""
            if len(cmd) >= 2 and cmd[1] == "predict":
                Path(cmd[cmd.index("--out") + 1]).write_text("mean\n1.5\n1.5\n")
                return 0, "", ""
            return 1, "", f"unexpected command: {cmd}"

        with tempfile.TemporaryDirectory() as td:
            with _patched_attrs(
                (_RUN_SUITE, "_ensure_rust_binary", lambda: Path("/tmp/fake-rust-gam")),
                (_RUN_SUITE, "run_cmd", _fake_run_cmd),
                (_RUN_SUITE, "_workspace_tempdir", _tempdir_factory(Path(td))),
                (
                    _RUN_SUITE,
                    "_rust_formula_for_scenario",
                    lambda *_args, **_kwargs: (
                        "gaussian",
                        "logratio ~ s(range, type=ps, knots=24)",
                    ),
                ),
            ):
                result = _RUN_SUITE.run_rust_scenario_cv(scenario, ds=ds, folds=folds)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evaluation"], "2-fold CV")
        self.assertEqual(result["n_folds"], 2)
        self.assertIn("[2-fold CV]", result["model_spec"])

    def test_main_does_not_schedule_rust_gam_for_survival_scenarios(self) -> None:
        seen = []

        def _fake_run_rust_scenario_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            seen.append(("rust_gam", kwargs.get("contender_name", "rust_gam")))
            return {
                "status": "ok",
                "scenario_name": "heart_failure_survival",
                "contender": "rust_gam",
                "model_spec": "5-fold CV",
                "evaluation": "cv",
            }

        def _fake_run_rust_gamlss_survival_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            seen.append(("survival", "rust_gamlss_survival"))
            return {
                "status": "ok",
                "scenario_name": "heart_failure_survival",
                "contender": "rust_gamlss_survival",
                "model_spec": "5-fold CV",
                "evaluation": "cv",
            }

        with tempfile.TemporaryDirectory() as td:
            scenario_path, out_path = _write_scenarios(Path(td), [{"name": "heart_failure_survival"}])
            with _patched_attrs(
                (_RUN_SUITE, "run_rust_scenario_cv", _fake_run_rust_scenario_cv),
                (_RUN_SUITE, "run_rust_gamlss_survival_cv", _fake_run_rust_gamlss_survival_cv),
                *_main_patches(
                    scenario_path,
                    out_path,
                    {
                        "rows": [{"time": 1.0, "event": 1.0, "x": 0.0}],
                        "features": ["x"],
                        "family": "survival",
                        "time_col": "time",
                        "event_col": "event",
                    },
                    lambda _scenario, contender: contender == "rust_gamlss_survival",
                ),
            ):
                _RUN_SUITE.main()

        self.assertNotIn(("rust_gam", "rust_gam"), seen)
        self.assertIn(("survival", "rust_gamlss_survival"), seen)

    def test_main_skips_flexible_variants_for_gaussian_scenarios(self) -> None:
        seen_rust = []
        seen_gamlss = []

        def _fake_run_rust_scenario_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            seen_rust.append(kwargs.get("contender_name", "rust_gam"))
            return {
                "status": "ok",
                "scenario_name": "lidar_semipar",
                "contender": kwargs.get("contender_name", "rust_gam"),
                "model_spec": "cv",
                "evaluation": "cv",
            }

        def _fake_run_rust_gamlss_scenario_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            seen_gamlss.append(kwargs.get("contender_name", "rust_gamlss"))
            return {
                "status": "ok",
                "scenario_name": "lidar_semipar",
                "contender": kwargs.get("contender_name", "rust_gamlss"),
                "model_spec": "cv",
                "evaluation": "cv",
            }

        with tempfile.TemporaryDirectory() as td:
            scenario_path, out_path = _write_scenarios(Path(td), [{"name": "lidar_semipar"}])
            with _patched_attrs(
                (_RUN_SUITE, "run_rust_scenario_cv", _fake_run_rust_scenario_cv),
                (_RUN_SUITE, "run_rust_gamlss_scenario_cv", _fake_run_rust_gamlss_scenario_cv),
                (_RUN_SUITE, "run_rust_gamlss_survival_cv", lambda *args, **kwargs: None),
                *_main_patches(
                    scenario_path,
                    out_path,
                    {
                        "rows": [{"y": 0.0, "range": 0.0}],
                        "features": ["range"],
                        "target": "y",
                        "family": "gaussian",
                    },
                    lambda _scenario, contender: contender in {"rust_gam", "rust_gamlss"},
                ),
            ):
                _RUN_SUITE.main()

        self.assertIn("rust_gam", seen_rust)
        self.assertNotIn("rust_gam_flexible", seen_rust)
        self.assertIn("rust_gamlss", seen_gamlss)
        self.assertNotIn("rust_gamlss_flexible", seen_gamlss)

    def test_main_skips_flexible_variants_for_survival_scenarios(self) -> None:
        seen_survival = []

        def _fake_run_rust_gamlss_survival_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            seen_survival.append(kwargs.get("contender_name", "rust_gamlss_survival"))
            return {
                "status": "ok",
                "scenario_name": "heart_failure_survival",
                "contender": kwargs.get("contender_name", "rust_gamlss_survival"),
                "model_spec": "cv",
                "evaluation": "cv",
            }

        with tempfile.TemporaryDirectory() as td:
            scenario_path, out_path = _write_scenarios(Path(td), [{"name": "heart_failure_survival"}])
            with _patched_attrs(
                (
                    _RUN_SUITE,
                    "run_rust_scenario_cv",
                    lambda *args, **kwargs: {
                        "status": "ok",
                        "scenario_name": "heart_failure_survival",
                        "contender": kwargs.get("contender_name", "rust_gam"),
                        "model_spec": "cv",
                        "evaluation": "cv",
                    },
                ),
                (_RUN_SUITE, "run_rust_gamlss_scenario_cv", lambda *args, **kwargs: None),
                (_RUN_SUITE, "run_rust_gamlss_survival_cv", _fake_run_rust_gamlss_survival_cv),
                *_main_patches(
                    scenario_path,
                    out_path,
                    {
                        "rows": [{"time": 1.0, "event": 1.0, "x": 0.0}],
                        "features": ["x"],
                        "time_col": "time",
                        "event_col": "event",
                        "family": "survival",
                    },
                    lambda _scenario, contender: contender
                    in {"rust_gam_sas", "rust_gamlss_survival"},
                ),
            ):
                _RUN_SUITE.main()

        self.assertEqual(seen_survival, ["rust_gamlss_survival"])

    def test_main_does_not_require_excluded_core_rust_contenders(self) -> None:
        seen = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss = _RUN_SUITE.run_rust_gamlss_scenario_cv
        orig_run_gamlss_marginal_slope = _RUN_SUITE.run_rust_gamlss_marginal_slope_cv
        orig_run_gamlss_survival = _RUN_SUITE.run_rust_gamlss_survival_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_datapoint_figures = _RUN_SUITE.generate_scenario_datapoint_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _unexpected_runner(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
                contender = kwargs.get("contender_name", "unknown")
                seen.append(contender)
                raise AssertionError(f"excluded contender scheduled: {contender}")

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                exclude_contenders = [
                    "rust_gam",
                    "rust_gam_flexible",
                    "rust_gamlss",
                    "rust_gamlss_flexible",
                    "rust_gamlss_marginal_slope",
                    "rust_gamlss_marginal_slope_aniso",
                    "r_gamlss",
                    "r_mgcv",
                    "r_mgcv_gaulss",
                    "r_gamboostlss",
                    "r_bamlss",
                    "r_brms",
                ]
                scenario_path.write_text(
                    json.dumps(
                        {
                            "scenarios": [
                                {
                                    "name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                                    "exclude_contenders": exclude_contenders,
                                }
                            ]
                        }
                    )
                )

                _RUN_SUITE.run_rust_scenario_cv = _unexpected_runner
                _RUN_SUITE.run_rust_gamlss_scenario_cv = _unexpected_runner
                _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = _unexpected_runner
                _RUN_SUITE.run_rust_gamlss_survival_cv = _unexpected_runner
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0, "pc1": 0.0}],
                    "features": ["pc1"],
                    "target": "y",
                    "family": "binomial",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                excluded = set(exclude_contenders)
                _RUN_SUITE._is_contender_enabled = lambda _scenario, contender: contender not in (
                    excluded
                    | {
                        "rust_gamlss_survival",
                        "rust_gamlss_survival_marginal_slope",
                        "r_mgcv_coxph",
                        "python_sksurv_rsf",
                        "python_sksurv_coxnet",
                        "python_lifelines_coxph_enet",
                        "r_glmnet_cox",
                        "python_sksurv_gb_coxph",
                        "python_sksurv_componentwise_gb_coxph",
                        "python_lifelines_weibull_aft",
                        "python_lifelines_lognormal_aft",
                        "python_xgboost_aft",
                    }
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.generate_scenario_datapoint_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
                payload = json.loads(out_path.read_text())
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_scenario_cv = orig_run_gamlss
            _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = orig_run_gamlss_marginal_slope
            _RUN_SUITE.run_rust_gamlss_survival_cv = orig_run_gamlss_survival
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.generate_scenario_datapoint_figures = orig_datapoint_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(seen, [])
        self.assertEqual(payload["results"], [])

    def test_main_skips_excluded_aniso_marginal_slope_contender(self) -> None:
        seen = []
        orig_run = _RUN_SUITE.run_rust_scenario_cv
        orig_run_gamlss = _RUN_SUITE.run_rust_gamlss_scenario_cv
        orig_run_gamlss_marginal_slope = _RUN_SUITE.run_rust_gamlss_marginal_slope_cv
        orig_parse_args = _RUN_SUITE.argparse.ArgumentParser.parse_args
        orig_dataset = _RUN_SUITE.dataset_for_scenario
        orig_folds = _RUN_SUITE.folds_for_dataset
        orig_assert_parity = _RUN_SUITE._assert_basis_parity_for_scenario
        orig_shared = _RUN_SUITE.build_shared_fold_artifacts
        orig_enabled = _RUN_SUITE._is_contender_enabled
        orig_figures = _RUN_SUITE.generate_scenario_figures
        orig_datapoint_figures = _RUN_SUITE.generate_scenario_datapoint_figures
        orig_zip = _RUN_SUITE.zip_figure_dir
        try:
            def _fake_run_rust_gamlss_scenario_cv(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
                return {
                    "status": "ok",
                    "scenario_name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                    "contender": kwargs.get("contender_name", "rust_gamlss"),
                    "model_spec": "5-fold CV",
                    "evaluation": "5-fold CV",
                }

            def _unexpected_marginal_slope(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
                contender = kwargs.get("contender_name", "unknown")
                seen.append(contender)
                raise AssertionError(f"excluded contender scheduled: {contender}")

            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                scenario_path = td_path / "scenarios.json"
                out_path = td_path / "results.json"
                scenario_path.write_text(
                    json.dumps(
                        {
                            "scenarios": [
                                {
                                    "name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                                    "exclude_contenders": ["rust_gamlss_marginal_slope_aniso"],
                                }
                            ]
                        }
                    )
                )

                _RUN_SUITE.run_rust_scenario_cv = lambda *args, **kwargs: {
                    "status": "ok",
                    "scenario_name": "geo_subpop16_margslope_aniso_duchon16d_k50",
                    "contender": kwargs.get("contender_name", "rust_gam"),
                    "model_spec": "5-fold CV",
                    "evaluation": "5-fold CV",
                }
                _RUN_SUITE.run_rust_gamlss_scenario_cv = _fake_run_rust_gamlss_scenario_cv
                _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = _unexpected_marginal_slope
                _RUN_SUITE.argparse.ArgumentParser.parse_args = lambda _self: SimpleNamespace(
                    scenarios=scenario_path,
                    out=out_path,
                    scenario_names=None,
                )
                _RUN_SUITE.dataset_for_scenario = lambda _scenario: {
                    "rows": [{"y": 0.0, "pc1": 0.0}],
                    "features": ["pc1"],
                    "target": "y",
                    "family": "binomial",
                }
                _RUN_SUITE.folds_for_dataset = lambda _ds: []
                _RUN_SUITE._assert_basis_parity_for_scenario = lambda *args, **kwargs: None
                _RUN_SUITE.build_shared_fold_artifacts = lambda *args, **kwargs: []
                enabled = {"rust_gam", "rust_gamlss"}
                _RUN_SUITE._is_contender_enabled = (
                    lambda scenario, contender: contender in enabled
                    and contender not in set(scenario.get("exclude_contenders", []))
                )
                _RUN_SUITE.generate_scenario_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.generate_scenario_datapoint_figures = lambda *_args, **_kwargs: []
                _RUN_SUITE.zip_figure_dir = lambda *_args, **_kwargs: None
                _RUN_SUITE.main()
        finally:
            _RUN_SUITE.run_rust_scenario_cv = orig_run
            _RUN_SUITE.run_rust_gamlss_scenario_cv = orig_run_gamlss
            _RUN_SUITE.run_rust_gamlss_marginal_slope_cv = orig_run_gamlss_marginal_slope
            _RUN_SUITE.argparse.ArgumentParser.parse_args = orig_parse_args
            _RUN_SUITE.dataset_for_scenario = orig_dataset
            _RUN_SUITE.folds_for_dataset = orig_folds
            _RUN_SUITE._assert_basis_parity_for_scenario = orig_assert_parity
            _RUN_SUITE.build_shared_fold_artifacts = orig_shared
            _RUN_SUITE._is_contender_enabled = orig_enabled
            _RUN_SUITE.generate_scenario_figures = orig_figures
            _RUN_SUITE.generate_scenario_datapoint_figures = orig_datapoint_figures
            _RUN_SUITE.zip_figure_dir = orig_zip

        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()

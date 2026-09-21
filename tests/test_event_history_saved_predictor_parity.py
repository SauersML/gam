"""A saved event-history predictor is one document for every surface (#2966).

`gam fit-events --save-model` and `EventHistoryModel.save` write the same Rust
document, and `gam forecast-events` and `gamfit.load` read it through the same
Rust code. Fresh processes on either side of the document forecast the same
held-out histories to the same bits: the survival, the expected counts, the
checked error of each, and how many posterior states were averaged.

The fixture is a risk-set-centred model with a latent atom, so every forecast
goes through the posterior average (#2964), and the document carries the
posterior directions and the reference law. Both tests fit the same cohort: a
fit of it takes about 80 s at 12 threads on the CLI and in gamfit alike, while
the seed-12 cohort had not finished after 420 s (probe 1258768). The CLI
forecasts with one Rayon thread and gamfit with its default, so bitwise
agreement is also thread independence.

The negative controls show that the surfaces read the document itself:
- a perturbed stored coefficient moves the forecast;
- a document of another version or kind, or with its reference profiles
  removed, is refused by name;
- the document holds the predictor's parts and no training record;
- the same document with no posterior directions is the plug-in at the mode,
  which differs from the average;
- a save that cannot be written fails `gam fit-events` before it writes its
  summary.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import gamfit

MARKS = {"visit": "recurrent", "disease": "once", "death": "terminal"}
# Prior-centred rates and loadings of one shared static frailty. The recurrent
# visits carry most of the information about a subject's frailty, which is what
# lets the evidence buy the atom.
RATES = {"visit": 0.8, "disease": 0.25, "death": 0.15}
LOADINGS = {"visit": 1.0, "disease": 1.2, "death": 0.9}
SCORE_EFFECT = 0.3
FOLLOW_UP = 4.0
HORIZONS_AFTER_EXIT = [1.0, 2.0]
# A risk-set-centred model forecasts only inside its reference law's interval,
# the training follow-up, so held-out histories end where their last horizon
# reaches it.
HELD_OUT_EXIT = FOLLOW_UP - max(HORIZONS_AFTER_EXIT)
# What a saved predictor holds: its schema, frozen bases, settings, posterior
# and reference law. Any other field would be a record the document must not
# carry.
PREDICTOR_FIELDS = {
    "mark_names",
    "mark_kinds",
    "covariate_names",
    "covariate_levels",
    "frozen_specs",
    "quadrature_order",
    "mesh_refinement",
    "time_scale",
    "gauss_hermite_order",
    "mode",
    "widths",
    "directions",
    "atoms",
    "rate_band",
    "held_rates",
    "reference",
}
FORECAST_FIELDS = ("horizons", "survival", "survival_error", "expected_counts", "expected_count_errors")


def _gam_binary() -> str:
    """Resolve the `gam` CLI the way the other CLI tests in tests/ do. There is
    no skip: an unbuilt CLI leaves this measurement undone, and it reads red."""
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (
        os.environ.get("GAM_BIN"),
        repo_root / "target" / "release" / "gam",
        repo_root / "target" / "debug" / "gam",
        shutil.which("gam"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise AssertionError(
        "[2966 H1] no `gam` CLI binary found (GAM_BIN, target/release/gam, target/debug/gam, PATH)"
    )


def _cli(arguments: list[str], one_thread: bool = True) -> subprocess.CompletedProcess:
    """Run the CLI in a fresh process, by default with a single Rayon thread."""
    environment = dict(os.environ, RAYON_NUM_THREADS="1") if one_thread else None
    return subprocess.run(arguments, capture_output=True, text=True, env=environment)


def _run(arguments: list[str], one_thread: bool = True) -> None:
    completed = _cli(arguments, one_thread)
    # Measurement precondition, printed before it is asserted.
    print(f"[2966 parity] {' '.join(arguments[1:2])} exited {completed.returncode}")
    assert completed.returncode == 0, (
        f"[2966 H2] {' '.join(arguments[:2])} exited {completed.returncode}: {completed.stderr}"
    )


def _simulate(n: int, seed: int, prefix: str, follow_up: float = FOLLOW_UP):
    """A recurrent ``visit``, a once-only ``disease`` and a terminal ``death``,
    whose log-rates share one standard-normal frailty through ``LOADINGS`` and a
    standard-normal score ``g``, with administrative censoring at
    ``follow_up``."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    frailty = rng.standard_normal(n)
    ids, times, marks, exits = [], [], [], []
    for i in range(n):
        sid = f"{prefix}{i}"
        rate = {
            mark: RATES[mark]
            * np.exp(LOADINGS[mark] * frailty[i] - 0.5 * LOADINGS[mark] ** 2 + SCORE_EFFECT * g[i])
            for mark in MARKS
        }
        t, at_risk, exit_ = 0.0, True, follow_up
        while True:
            active = [mark for mark in MARKS if mark != "disease" or at_risk]
            total = sum(rate[mark] for mark in active)
            t -= np.log(rng.uniform()) / total
            if t >= follow_up:
                break
            mark = active[rng.choice(len(active), p=[rate[m] / total for m in active])]
            ids.append(sid)
            times.append(t)
            marks.append(mark)
            if mark == "death":
                exit_ = t
                break
            if mark == "disease":
                at_risk = False
        exits.append(exit_)
    names = [f"{prefix}{i}" for i in range(n)]
    subjects = pd.DataFrame({"id": names, "entry": 0.0, "exit": exits})
    events = pd.DataFrame({"id": ids, "time": times, "mark": marks})
    covariates = pd.DataFrame({"id": names, "start": 0.0, "g": g})
    return subjects, events, covariates


def _held_out(count: int, seed: int):
    """``count`` histories censored at ``HELD_OUT_EXIT``, from a cohort
    simulated apart from the training cohort, so each has a future to forecast
    inside the reference interval."""
    subjects, events, covariates = _simulate(4 * count, seed, "heldout", follow_up=HELD_OUT_EXIT)
    died = set(events.loc[events["mark"] == "death", "id"])
    keep = [sid for sid in subjects["id"] if sid not in died][:count]
    # Measurement precondition, printed before it is asserted.
    print(f"[2966 parity] censored held-out histories: {len(keep)} of {count}")
    assert len(keep) == count, f"[2966 H3] only {len(keep)} censored held-out histories"
    return (
        subjects[subjects["id"].isin(keep)].reset_index(drop=True),
        events[events["id"].isin(keep)].reset_index(drop=True),
        covariates[covariates["id"].isin(keep)].reset_index(drop=True),
    )


def _write_tables(directory: Path, stem: str, subjects, events, covariates) -> dict[str, str]:
    """The tables as CSV with 17 significant digits, so the CLI parses exactly
    the doubles Python holds."""
    paths = {}
    for name, frame in (("subjects", subjects), ("events", events), ("covariates", covariates)):
        path = directory / f"{stem}_{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        paths[name] = str(path)
    return paths


def _forecast_arguments(gam: str, model_path: Path, paths: dict[str, str], out: Path) -> list[str]:
    return [
        gam,
        "forecast-events",
        "--model",
        str(model_path),
        "--subjects",
        paths["subjects"],
        "--events",
        paths["events"],
        "--covariates",
        paths["covariates"],
        "--horizons-after-exit",
        ",".join(repr(h) for h in HORIZONS_AFTER_EXIT),
        "--out",
        str(out),
    ]


def _cli_forecasts(gam: str, model_path: Path, paths: dict[str, str], out: Path) -> dict[str, dict]:
    _run(_forecast_arguments(gam, model_path, paths, out))
    document = json.loads(out.read_text())
    return {entry["id"]: entry for entry in document["forecasts"]}


def _gamfit_forecasts(predictor, subjects, events, covariates) -> dict[str, dict]:
    forecasts = {}
    for row in subjects.itertuples(index=False):
        own = events[events["id"] == row.id]
        history = [(float(t), str(m)) for t, m in zip(own["time"], own["mark"])]
        g = float(covariates.loc[covariates["id"] == row.id, "g"].iloc[0])
        horizons = [float(row.exit) + h for h in HORIZONS_AFTER_EXIT]
        forecasts[row.id] = predictor.forecast_history(
            float(row.entry), float(row.exit), history, {"g": g}, horizons
        )
    return forecasts


def _hex(values) -> list:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [float(v).hex() for v in array]
    return [[float(v).hex() for v in row] for row in array]


def _assert_same_bits(label: str, cli: dict[str, dict], python: dict[str, dict]) -> None:
    assert set(cli) == set(python), f"{label}: the surfaces forecast different subjects"
    for sid, forecast in python.items():
        for key in FORECAST_FIELDS:
            assert _hex(cli[sid][key]) == _hex(forecast[key]), f"{label}: subject {sid}: {key} differs"
        assert int(cli[sid]["posterior_evaluations"]) == int(forecast["posterior_evaluations"]), (
            f"{label}: subject {sid}: posterior_evaluations differs"
        )
        print(
            f"[2966 parity] {label} subject {sid}: posterior_evaluations={forecast['posterior_evaluations']} "
            f"survival={[float(v).hex() for v in forecast['survival']]}"
        )


def _assert_averaged(label: str, forecasts: dict[str, dict]) -> None:
    """Every forecast was averaged over more than one posterior state."""
    for sid, forecast in forecasts.items():
        assert int(forecast["posterior_evaluations"]) > 1, (
            f"{label}: subject {sid} evaluated {forecast['posterior_evaluations']} posterior states"
        )


def _moves(a: dict, b: dict) -> float:
    """The largest amount by which two forecasts of the same history differ
    beyond both checked errors. Positive means they differ."""
    largest = -np.inf
    for value, error in (("survival", "survival_error"), ("expected_counts", "expected_count_errors")):
        gap = np.abs(np.asarray(a[value], dtype=float) - np.asarray(b[value], dtype=float))
        bar = np.asarray(a[error], dtype=float) + np.asarray(b[error], dtype=float)
        largest = max(largest, float(np.max(gap - bar)))
    return largest


def _cli_refuses(label: str, gam: str, model_path: Path, paths: dict[str, str], out: Path, word: str) -> None:
    completed = _cli(_forecast_arguments(gam, model_path, paths, out))
    assert completed.returncode != 0, f"{label}: the CLI forecast from {model_path.name}, which it must refuse"
    assert word in completed.stderr.lower(), f"{label}: the CLI's refusal names no {word!r}: {completed.stderr}"
    assert not out.exists(), f"{label}: a refused document still wrote forecasts"
    print(f"[2966 parity] CLI refused {model_path.name}: {completed.stderr.strip()[:200]}")


def _gamfit_refuses(label: str, model_path: Path, word: str) -> None:
    try:
        gamfit.load(model_path)
    except Exception as error:  # noqa: BLE001 - the refusal's message is what is asserted
        assert word in str(error).lower(), f"{label}: gamfit's refusal names no {word!r}: {error}"
        print(f"[2966 parity] gamfit refused {model_path.name}: {str(error)[:200]}")
        return
    raise AssertionError(f"{label}: gamfit loaded {model_path.name}, which it must refuse")


def _write_document(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document, indent=2))
    return path


def _fit_arguments(
    gam: str, training: dict[str, str], summary: Path, model: Path, reference: bool = True
) -> list[str]:
    return [
        gam,
        "fit-events",
        "--subjects",
        training["subjects"],
        "--events",
        training["events"],
        "--covariates",
        training["covariates"],
        "--formula",
        "g",
        "--marks",
        "visit:recurrent,disease:once,death:terminal",
        *(["--reference-row", "0"] if reference else []),
        "--out",
        str(summary),
        "--save-model",
        str(model),
    ]


def test_a_predictor_the_cli_saves_forecasts_the_same_bits_in_gamfit(tmp_path: Path) -> None:
    subjects, events, covariates = _simulate(120, seed=11, prefix="s")
    training = _write_tables(tmp_path, "training", subjects, events, covariates)
    model_path = tmp_path / "predictor.json"
    summary_path = tmp_path / "summary.json"
    gam = _gam_binary()
    _run(_fit_arguments(gam, training, summary_path, model_path), one_thread=False)
    summary = json.loads(summary_path.read_text())
    print(f"[2966 parity] CLI fit: rank={summary['rank']} reference_certificate={summary.get('reference_certificate')}")
    assert summary["rank"] >= 1, f"[2966 P3] the fixture's evidence bought no atom: {summary['rank_path']}"
    assert summary.get("reference_certificate") is not None, "[2966 P4] the fit is not risk-set centred"
    held_out = _held_out(2, seed=21)
    paths = _write_tables(tmp_path, "held_out", *held_out)
    cli = _cli_forecasts(gam, model_path, paths, tmp_path / "cli.json")
    loaded = gamfit.load(model_path)
    assert isinstance(loaded, gamfit.EventHistoryPredictor), f"[2966 P5] gamfit.load returned {type(loaded)}"
    assert loaded.mark_kinds == MARKS, f"[2966 P6] the loaded mark kinds are {loaded.mark_kinds}"
    python = _gamfit_forecasts(loaded, *held_out)
    _assert_same_bits("[2966 P7] cli-saved", cli, python)
    _assert_averaged("[2966 P8] cli-saved", python)

    # A requested save is part of the command's output: one that cannot be
    # written fails the command, naming the file, before the summary is written.
    # Whether a save can be written does not depend on the model, so a small
    # prior-centred cohort carries this control instead of a second centred fit.
    small = _write_tables(tmp_path, "small", *_simulate(40, seed=13, prefix="p"))
    unwritable = tmp_path / "missing" / "predictor.json"
    refused_summary = tmp_path / "refused_summary.json"
    completed = _cli(_fit_arguments(gam, small, refused_summary, unwritable, reference=False), one_thread=False)
    print(
        f"[2966 parity] fit-events saving to {unwritable} exited {completed.returncode}: "
        f"{completed.stderr.strip()[:200]}"
    )
    assert completed.returncode != 0 and str(unwritable) in completed.stderr, (
        f"[2966 P29] a save to {unwritable} did not fail the command by name: {completed.stderr}"
    )
    assert not refused_summary.exists(), "[2966 P30] a failed save still wrote the summary"


def test_a_predictor_gamfit_saves_forecasts_the_same_bits_in_the_cli(tmp_path: Path) -> None:
    subjects, events, covariates = _simulate(120, seed=11, prefix="s")
    model = gamfit.fit_event_history(
        subjects, events, covariates, "g", marks=MARKS, reference_profiles=[0]
    )
    print(f"[2966 parity] gamfit fit: rank={model.rank} reference_certificate={model.reference_certificate}")
    assert model.rank >= 1, f"[2966 P9] the fixture's evidence bought no atom: {model.rank_path}"
    assert model.reference_certificate is not None, "[2966 P10] the fit is not risk-set centred"
    model_path = tmp_path / "predictor.json"
    model.save(model_path)
    held_out = _held_out(2, seed=22)
    paths = _write_tables(tmp_path, "held_out", *held_out)
    gam = _gam_binary()
    cli = _cli_forecasts(gam, model_path, paths, tmp_path / "cli.json")
    in_memory = _gamfit_forecasts(model, *held_out)
    _assert_same_bits("[2966 P11] gamfit-saved", cli, in_memory)
    _assert_averaged("[2966 P12] gamfit-saved", in_memory)
    reloaded = gamfit.load(model_path)
    _assert_same_bits("[2966 P13] gamfit-reloaded", cli, _gamfit_forecasts(reloaded, *held_out))
    # The reloaded predictor saves the document it was read from, byte for byte.
    resaved = tmp_path / "resaved.json"
    reloaded.save(resaved)
    assert resaved.read_bytes() == model_path.read_bytes(), "[2966 P14] the re-saved document differs"

    # A subject with no history, from a covariate path alone: `gam
    # forecast-population` and the reloaded predictor's population_forecast
    # give the same bits, averaged over the posterior.
    population_path = tmp_path / "population_covariates.csv"
    pd.DataFrame({"start": [0.0], "g": [0.4]}).to_csv(population_path, index=False, float_format="%.17g")
    population_out = tmp_path / "population_cli.json"
    _run(
        [
            gam,
            "forecast-population",
            "--model",
            str(model_path),
            "--covariates",
            str(population_path),
            "--start",
            repr(0.0),
            "--horizons",
            ",".join(repr(h) for h in HORIZONS_AFTER_EXIT),
            "--out",
            str(population_out),
        ]
    )
    population_cli = {"population": json.loads(population_out.read_text())["forecast"]}
    population_python = {
        "population": reloaded.population_forecast({"g": 0.4}, 0.0, HORIZONS_AFTER_EXIT)
    }
    _assert_same_bits("[2966 P1] population", population_cli, population_python)
    _assert_averaged("[2966 P2] population", population_python)

    document = json.loads(model_path.read_text())
    # The document holds the predictor's parts and no training record.
    text = model_path.read_text()
    named = [sid for sid in subjects["id"] if f'"{sid}"' in text]
    assert not named, f"[2966 P17] the document names training subjects {named[:5]}"
    assert set(document["model"]) == PREDICTOR_FIELDS, f"[2966 P15] document fields {sorted(document['model'])}"
    grid, profiles = document["model"]["reference"]
    # Measurement preconditions of the controls below, printed before they are
    # asserted. ndarray serialises a matrix as {"v", "dim", "data"}.
    print(
        f"[2966 parity] saved reference profiles {profiles['dim']}, "
        f"posterior directions {len(document['model']['directions'])}"
    )
    assert profiles["dim"][0] == 1, f"[2966 P16] one reference stratum, one profile row: {profiles['dim']}"
    assert document["model"]["directions"], "[2966 P18] the saved posterior has no directions"

    # A perturbed stored coefficient: both surfaces read the change, and it moves
    # the forecast beyond both checked errors.
    perturbed = json.loads(json.dumps(document))
    perturbed["model"]["mode"][0] += 0.5
    perturbed_path = _write_document(tmp_path / "perturbed.json", perturbed)
    perturbed_cli = _cli_forecasts(gam, perturbed_path, paths, tmp_path / "perturbed_cli.json")
    perturbed_python = _gamfit_forecasts(gamfit.load(perturbed_path), *held_out)
    _assert_same_bits("[2966 P19] perturbed", perturbed_cli, perturbed_python)
    for sid in in_memory:
        moved = _moves(in_memory[sid], perturbed_python[sid])
        print(f"[2966 parity] perturbed subject {sid}: largest change beyond both checked errors {moved:.3e}")
        assert moved > 0.0, f"[2966 P20] subject {sid}: a perturbed coefficient left the forecast within its checked errors"

    # The same document with no posterior directions is the plug-in at the mode:
    # one parameter state, and a forecast the average differs from.
    plug_in = json.loads(json.dumps(document))
    plug_in["model"]["directions"] = []
    plug_in_path = _write_document(tmp_path / "plug_in.json", plug_in)
    plug_in_cli = _cli_forecasts(gam, plug_in_path, paths, tmp_path / "plug_in_cli.json")
    plug_in_python = _gamfit_forecasts(gamfit.load(plug_in_path), *held_out)
    _assert_same_bits("[2966 P21] plug-in", plug_in_cli, plug_in_python)
    for sid in in_memory:
        assert int(plug_in_python[sid]["posterior_evaluations"]) == 1, (
            f"[2966 P22] subject {sid}: the plug-in evaluated {plug_in_python[sid]['posterior_evaluations']} states"
        )
        gap = _moves(in_memory[sid], plug_in_python[sid])
        print(f"[2966 parity] plug-in subject {sid}: largest gap to the average beyond both checked errors {gap:.3e}")
        assert gap > 0.0, f"[2966 P23] subject {sid}: the plug-in agrees with the average within both checked errors"

    # Documents the surfaces must refuse, by name.
    older = json.loads(json.dumps(document))
    older["version"] = 0
    older_path = _write_document(tmp_path / "older.json", older)
    _cli_refuses("[2966 P24] older", gam, older_path, paths, tmp_path / "older_cli.json", "version")
    _gamfit_refuses("[2966 P25] older", older_path, "version")
    other_kind = json.loads(json.dumps(document))
    other_kind["kind"] = "joint"
    other_kind_path = _write_document(tmp_path / "other_kind.json", other_kind)
    _cli_refuses("[2966 P26] other kind", gam, other_kind_path, paths, tmp_path / "other_kind_cli.json", "kind")
    no_profiles = json.loads(json.dumps(document))
    no_profiles["model"]["reference"] = [grid]
    no_profiles_path = _write_document(tmp_path / "no_profiles.json", no_profiles)
    _cli_refuses("[2966 P27] no profiles", gam, no_profiles_path, paths, tmp_path / "no_profiles_cli.json", "model document")
    _gamfit_refuses("[2966 P28] no profiles", no_profiles_path, "model document")

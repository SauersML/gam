# Packaging / install / footprint / Python-surface audit: gamfit vs pyGAM

Environment: bvenv (gamfit 0.1.267 wheel from PyPI, pyGAM 0.12.0). Extra venvs: v310 (numpy 1.26.4), v313 (numpy 2.5.3), v314 (3.14.0rc2), v313t (free-threaded).
Scripts and raw outputs are in `scratchpad/audit/packaging/`. No cargo builds were run and nothing under /home/user/gam was modified.

## Measurements

| Metric | gamfit 0.1.267 | pyGAM 0.12.0 |
|---|---|---|
| Wheel size | 29.5–36.5 MB (manylinux x86_64 34.33 MB, win 36.51 MB); sdist 17.16 MB | 0.08 MB |
| Installed (package itself) | 79 MB (`_rust.abi3.so` 79.5 MB, stripped) + 1.1 MB dist-info (986 KB CycloneDX SBOM) | 0.7 MB |
| Installed with required deps | numpy only: ~80 MB + numpy | scipy (99 MB + 30 MB libs) + progressbar2 + python-utils + typing_extensions: ~130 MB + numpy |
| Required deps | numpy>=1.26 | numpy, scipy<1.17, progressbar2 |
| Heavy modules pulled in by `import` | none (no torch, jax, pandas, matplotlib, sklearn, scipy); 323 modules | scipy.stats, progressbar; 870 modules |
| `-X importtime` cumulative | 178.6 ms (numpy ~55 ms) | 485.7 ms |
| Cold wall time, 5 runs (loaded box) | 285–327 ms (numpy alone: 146–197) | 673–904 ms |
| `gamfit.sklearn` import | 1014–1442 ms (imports sklearn) | n/a |
| RSS after import | 56.5 MB (numpy alone: 33.6) | 111.3 MB |
| Threads after import | 9 (numpy alone: 4): 4 rayon workers + gam-process-monitor | 4 |
| Public names | 331 in `__all__`, 341 in dir, 16 submodules | 12 |
| `fit` / estimator params | `gamfit.fit`: 39 (HEAD: 40 kw-only) | LinearGAM: 8 |
| Model public methods | 41 | 15 |
| py.typed | yes; pyright verifytypes 85% (1104 known / 98 ambiguous / 97 unknown of 1299) | no (mypy: import-untyped) |
| pickle / deepcopy of fitted model | FAILS | works |
| multiprocessing "fork" child fit | HANGS | works (2.5 s) |
| Warnings on a basic fit | 1 GamInferenceWarning per smooth, plus raw Rust log lines on stderr | none (summary() emits a "KNOWN BUG p-values" UserWarning) |
| Python matrix actually working | abi3 wheel verified on 3.10 (numpy 1.26.4), 3.13, 3.14rc2; no 3.13t wheel | pure Python |
| Declared classifiers | 3.10–3.13 | up to 3.14 |
| Wheel platforms | manylinux2014 x86_64/aarch64, musllinux_1_2 x86_64/aarch64, macOS x86_64/arm64, win x64 | any |
| conda-forge | 404 | pygam 0.12.0 |
| License | AGPL-3.0-or-later | Apache-2.0 |

## Findings

### PKG-01: `import gamfit` makes every later fork-based fit deadlock (bug, high)
- **Evidence.**
  - `fork_noparent.py`: the parent only runs `import gamfit`, then `multiprocessing.get_context("fork").Pool(2).map(fit_one, ...)`. The pool timed out at 45 s with 0.47 s user CPU.
  - `fork_min.py`: `os.fork()` followed by `fit()` in the child hangs in `rust_module().fit_table` (_api.py:1291 in the wheel).
  - The spawn context works (7 s). pyGAM under fork works (2.5 s).
  - "fork" is the default start method on Linux for Python 3.10–3.13, so joblib, `Pool`, and sklearn `n_jobs` users with fork backends are affected.
- **Root cause.** The `#[pymodule]` init (crates/gam-pyffi/src/manifold/geometry_ffi.rs ~4489–4502) calls `gam::init_parallelism()`. That function (src/lib.rs:108–137) eagerly runs `rayon::ThreadPoolBuilder::new()...build_global()` and also starts `gam_runtime::process_monitor::start()`. Worker threads do not survive a fork, and no at-fork handler exists (no pthread_atfork or register_at_fork anywhere).
- **Fix.** Do not build the pool at import. Keep a per-process pool behind a pid-checked lazy cell, create it on first FFI entry, and run work under `pool.install(...)`. A child then sees a pid mismatch and builds a fresh pool. Alternatively, `os.register_at_fork(after_in_child=_rust._reset_after_fork)` can rebuild the pool. Start the monitor lazily, or not at all by default (see PKG-03). Add a regression test that forks after import and fits.
- **Files:** src/lib.rs, crates/gam-pyffi/src/manifold/geometry_ffi.rs, crates/gam-runtime/src/process_monitor.rs, a new tests/python fork test.
- **Size:** M.

### PKG-02: Fitted `Model` cannot be pickled or deep-copied (gap, high)
- **Evidence.** `pickle.dumps(model)` and `copy.deepcopy(model)` both raise `TypeError: cannot pickle 'builtins._FittedModel' object` (pickle_check.py). pyGAM models round-trip.
- **Consequences.** A fitted model cannot be sent through joblib/multiprocessing, cached, or cloned by sklearn after fit.
- **Cause.** crates/gam-pyffi/src/model/model_ffi.rs:79 `#[pyclass(name = "_FittedModel", frozen)]` has no `__reduce__`/`__getstate__`, and no `__reduce__` exists anywhere in gamfit or gam-pyffi. The `module=` attribute is missing, so the class also reports as `builtins._FittedModel`.
- **Related.** Many research pyclasses declare `module = "gam_pyffi._rust"`, which is not importable, so they cannot be pickled either.
- **Fix.** Implement `__reduce__` on `Model`, or `__getnewargs__`/`__getstate__` on `_FittedModel`, returning `(gamfit.loads, (self.dumps_bytes(),))` via the existing save/load serializer. Set `module = "gamfit._rust"` on all pyclasses. Add a pickle round-trip test, including for sklearn estimators.
- **Files:** model_ffi.rs:79, gamfit/_model.py, the other `#[pyclass(module=...)]` sites.
- **Size:** S.

### PKG-03: Raw Rust log lines go to stderr during ordinary fits, with no public control (bug, med)
- **Evidence.** A two-smooth gaussian fit in 0.1.267 printed `[0s] [HGB] target_mse below mandatory floors...` (twice) and `[0s] [OUTER] ARC cost-stall STUCK (NOT a flat valley)...` (stderr_check.py).
- **Status at HEAD.** crates/gam-solve/src/rho_optimizer/bridges.rs:2196 still has `log::warn!("[OUTER] cost-stall STUCK (NOT a flat valley)...")`. The HGB line is gone.
- **Mechanism.** `progress_log::init_logging()` installs a stderr logger at Warn during module init. The process monitor wakes every 60 s and emits STALL warns for phases older than 120 s (process_monitor.rs:22,32,181,256), which reach Jupyter and log collectors.
- **No off switch.** The only control is the private `gamfit._rust.set_log_level` (geometry_ffi.rs:4476). Python `logging` cannot filter these lines.
- **Fix.** Demote internal optimizer diagnostics to `debug!`/`trace!`. Anything the user must see becomes a structured note surfaced through the existing inference-warning path. Route Rust `log` through `pyo3-log` into the `gamfit` logger so standard `logging` config applies, then delete the private setter. Make the process monitor opt-in, enabled by an env var for development only.
- **Files:** crates/gam-pyffi/src/manifold/geometry_ffi.rs (init), progress_log, bridges.rs:2196, process_monitor.rs.
- **Size:** S–M.

### PKG-04: Every default `s(x)` fit emits a GamInferenceWarning, and in 0.1.267 it points at gamfit internals (bug, med)
- **Evidence.** `default_warn.py`: `fit(data, "y ~ s(x)")` with default filters emits "Automatically set N internal knots ... rule: clamp(unique/4, 4..8)" once per smooth. The source is crates/gam-terms/src/term_builder.rs:2839–2846.
- **Why it matters.** A warning on the happy path trains users to ignore warnings.
- **Stacklevel.** In the wheel the warning is attributed to `_api.py:693` (the kwarg-typo decorator's `return fn(*args, **kwargs)`), because `_warnings.py:29` uses a fixed `stacklevel=3`. HEAD removed the decorator from `fit` (_api.py:668), so this part is fixed but unreleased.
- **Fix.** Record the auto-knot choice in `model.notes`/summary rather than warning. Reserve warnings for conditions the user must act on. Compute stacklevel dynamically: walk frames until one is outside the `gamfit` package.
- **Files:** term_builder.rs:2839–2846, gamfit/_warnings.py:29.
- **Size:** S.

### PKG-05: `mypy --strict` rejects `gamfit.fit` / `gamfit.load`; no stub for `_rust`; key returns are `Any` (bug, med)
- **Evidence.** `user_typed.py` under `mypy --strict` reports `Module "gamfit" does not explicitly export attribute "fit"`, and the same for `load` and `adjudicate_atom_shape`.
- **Cause.** `__all__ = _build_public_api()` (gamfit/__init__.py:488) is computed at runtime, so mypy cannot see it, and the imports are `from ._api import fit` without the `as fit` re-export form.
- **Other typing issues.**
  - pyright verifytypes gives 85% completeness with 211 errors (verifytypes.txt), mostly torch submodules and unknown `_rust` types.
  - There is no `gamfit/_rust.pyi`.
  - `Model.predict(...) -> Any` (_model.py:1464) and `load(...) -> Any`.
- **Fix.**
  - Replace the dynamic `__all__` with a literal list and add a unit test asserting it equals the computed set.
  - Ship `_rust.pyi` for the FFI surface actually used by the wrapper.
  - Type `predict` (ndarray or a Prediction dataclass) and `load` (-> Model).
  - Add `mypy --strict` over a user-style snippet to CI.
- **Files:** gamfit/__init__.py:488, gamfit/_model.py:1464, a new gamfit/_rust.pyi, CI.
- **Size:** S–M.

### PKG-06: Top-level namespace is overloaded (331 names, ~220 SAE/manifold/research) with colliding names (gap, med)
- **Evidence (names.txt).** A heuristic split gives ~50 core GAM names, 62 exception classes and ~220 SAE, manifold, topology or dictionary names. pyGAM exports 12.
- **Collisions.**
  - Four distinct `Sphere`s: `gamfit.Sphere` (smooth.Sphere), `TopologySphere` (topology.Sphere), `SphereManifold` (Rust), `manifolds.Sphere`.
  - `Circle`/`CircleManifold` and `Torus`/`TorusManifold`.
  - `Smooth` vs `SmoothSpec`.
- **Broken core-looking name.** `gamfit.plot` is the SAE atom plotter: `gamfit.plot(model)` raises "TypeError atom must expose the SAE atom fields". The GAM plot is `Model.plot`.
- **Leak.** `pathlib.Path` shows up in `dir(gamfit)`.
- **Import cost.** Research modules are imported eagerly: _sae_spectral 13.8 ms, _sparse_dictionary 5 ms, _select_topology 4 ms.
- **Fix.**
  - Keep a curated core at top level: fit, load, Model, formula terms, families, summary/diagnostics, the sklearn entry, and a single exception base with its core subclasses.
  - Move SAE, manifold, topology and dictionary exports to `gamfit.sae` / `gamfit.manifold` namespaces, loaded lazily via module `__getattr__`.
  - Delete the duplicate aliases rather than keeping compatibility shims.
- **Files:** gamfit/__init__.py and the research submodules' import sites.
- **Size:** M–L.

### PKG-07: PyPI project storage is at 8.29 of 10 GB, with ~7 releases of headroom (gap, high)
- **Evidence.** The PyPI JSON API lists 69 retained releases (0.1.174 onward) totalling 8.29 GB, at ~246 MB per full release (8 wheels plus the 17 MB sdist).
- **Risk.** The default per-project limit is 10 GB. I could not verify whether a raise was granted. At the current cadence, uploads start failing in about 7 releases, and the release job uploads without a pre-check.
- **Fix.**
  - Request a size limit increase now.
  - Prune superseded pre-releases.
  - Cut per-release size: move the SAE/manifold research code into a separate extension or package (e.g. `gamfit-sae`) so the core `.so` shrinks from 79.5 MB, and exclude test data bins and `tests.rs` from the sdist (`[tool.maturin] exclude`).
  - Add a CI guard that fails before upload when projected usage exceeds 90%.
- **Files:** pyproject.toml ([tool.maturin] include/exclude), .github/workflows/pypi-wheels.yml, the crate split.
- **Size:** M.

### PKG-08: Built wheels are never installed and tested; Python tests run only on Linux 3.11/3.12 (gap, med)
- **Evidence.**
  - .github/workflows/pypi-wheels.yml builds 8 targets plus the sdist and uploads with twine (~lines 413–431) without installing any wheel.
  - The Python workflows pin 3.11 (python-contracts.yml:122, test.yml:1244/1530, validate-one.yml:132) or 3.12 (test-census, docs, cross-check).
  - The declared floor (3.10 with numpy 1.26) and 3.13/3.14 are never tested, nor are macOS, Windows or musllinux.
  - I verified by hand that 3.10 with numpy 1.26.4, 3.13 and 3.14rc2 work today (smoke.py), but nothing prevents a regression.
- **Fix.** Add a post-build matrix job: {each wheel artifact} × {3.10 with numpy==1.26.*, 3.13, 3.14}. Each job installs the wheel into a fresh venv from outside the repo and runs smoke.py: gaussian and binomial fit, predict, save/load, pickle once PKG-02 lands, and a fork test once PKG-01 lands. Gate the upload on it.
- **Files:** .github/workflows/pypi-wheels.yml.
- **Size:** S.

### PKG-09: Distribution coverage gaps (gap, low)
- **3.14.** Not in the classifiers, although the abi3 wheel works on 3.14rc2. pyGAM declares 3.14.
- **Free-threaded builds.** There is no free-threaded wheel: an `--only-binary` install on 3.13t is unsatisfiable, even though the module declares `gil_used=false`.
- **conda-forge.** No gamfit feedstock (404); pygam is there.
- **Other platforms.** No win-arm64, ppc64le, s390x or Pyodide wheels. These users fall back to the sdist, which needs the pinned Rust 1.97.1 toolchain (rust-toolchain.toml).
- **Fix.** Add the 3.14 classifier and a cp313t/cp314t wheel job. Create a conda-forge feedstock after PKG-07 slims the sdist. Win-arm64 is optional.
- **Files:** pyproject.toml, pypi-wheels.yml.
- **Size:** S (classifier, cp3xt) / M (conda-forge).

### PKG-10: Core-path Python logic census (SPEC violations, small) (gap, low)
The Python core path (`_api.py`, `_model.py`, `sklearn.py`, `_diagnostics.py`, `__init__.py`) contains no `np.linalg`, scipy, einsum or exp/log math; the only einsum, at _model.py:59, is in a docstring. The remaining logic:

| Location | Logic in Python | Fix |
|---|---|---|
| gamfit/_api.py:3677 | `_DEFAULT_BASIS_K = 10` default for the standalone basis helpers | Use one Rust source for the default K. The paths disagree today: `gamfit.bspline_basis(x)` gives 14 columns while the formula `s(x)` gives 8 internal knots (12 columns). |
| gamfit/_api.py:3722–3797 | `_resolve_knots` / `_periodic_uniform_grid` build periodic knot grids with `np.linspace` | Export an `auto_periodic_knots_1d` from Rust, next to `auto_knots_1d` / `auto_centers_1d`. |
| gamfit/sklearn.py:352 | `np.clip(posterior_mean, 0, 1)` in `predict_proba` | Rust should return valid probabilities; drop the clip, or assert. |
| gamfit/sklearn.py:449 and gamfit/_diagnostics.py:196 | `train_prev = np.mean(observed)` before calling `classification_metrics` | Compute the prevalence inside the Rust metric. |

- **Size:** S overall. The torch/ submodule (`_reml.py` 974 lines, `fit.py` 1011 lines, `torch.linalg.svd` at fit.py:422) falls under SPEC's PyTorch-interop exception and is not counted here.

### PKG-11: `fit()` exposes 39–40 keyword knobs, including solver tolerances (gap, low–med)
- **Evidence.** `inspect.signature(gamfit.fit)` shows 39 params. At HEAD there are 40 kw-only params, including `outer_tol`, `inner_tol`, `persistent_warm_start_root` and `config`. pyGAM's `LinearGAM` has 8.
- **SPEC conflict.** SPEC says no magic knobs, and a fit only comes from a converged optimization, so user-tunable tolerances contradict it.
- **Fix.** Delete `outer_tol`/`inner_tol` (Rust owns convergence). Fold `persistent_warm_start_root` into an internal cache. Audit the remaining kwargs against "unnecessary options are deleted".
- **Files:** gamfit/_api.py (fit signature) and the matching FFI args.
- **Size:** M.

## Already better than pyGAM (keep)
- **AB-1. Dependencies.** gamfit requires only numpy. pyGAM also requires scipy and progressbar2. Total install is ~80 MB vs ~130 MB with deps.
- **AB-2. Import cost.** No heavy imports at import time; torch and sklearn are lazy or opt-in. Import takes 179 ms vs 486 ms (importtime), and RSS is 56 MB vs 111 MB.
- **AB-3. Typing.** py.typed is shipped with 85% type completeness; pyGAM has none.
- **AB-4. Thin Python layer.** The Python layer is essentially thin: no linear algebra on the core path. pyGAM does its math in Python and scipy.
- **AB-5. Pure-Rust binary.** abi3-py310 wheels with a fat-LTO, stripped `release-pypi` profile, `locked = true`, and an SBOM, covering 8 targets including musllinux and aarch64.

## pyGAM slop to avoid
- **SLOP-1.** `pyGAM.summary()` emits `UserWarning: KNOWN BUG: p-values computed in this summary are likely much smaller than they should be` and prints further warnings. gamfit must never ship known-wrong statistics behind a warning.
- **SLOP-2.** A progressbar2 runtime dependency just for progress output. gamfit should likewise keep progress and diagnostics out of stdout/stderr by default (see PKG-03).

## License (competitive factor, user's decision)
gamfit is AGPL-3.0-or-later and pyGAM is Apache-2.0. For many corporate users AGPL is a barrier to adoption, independent of technical merit. This is noted for the user to decide on; no change is proposed.

## Incidental (off-axis, for the robustness auditor)
- Noiseless or near-noiseless gaussian responses fail. `fit({'x': linspace, 'y': sin(6x) + sd*noise}, 'y ~ s(x)')` with sd=0 or 1e-6 raises `IntegrationError: smoothing cubature has no positive-width proposal in the resolved domain` (noiseless.py). sd=1e-3 works, and pyGAM fits the sd=0 case (edof 13.0).

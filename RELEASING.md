# Releasing gamfit to PyPI

`pypi-wheels.yml` is the only path to PyPI. It is dispatched by hand with an
exact source commit and a scope, builds every wheel and the sdist, installs
each wheel into an empty virtualenv and fits with it
(`scripts/wheel_smoke.py`), and only then uploads the whole set with
`twine upload --skip-existing`.

## What one release puts on PyPI

For every platform family (manylinux and musllinux on x86_64 and aarch64,
macOS x86_64 and arm64, Windows x64):

* one `cp310-abi3` wheel, which every GIL CPython from 3.10 on installs;
* one wheel per free-threaded CPython the classifiers advertise
  (`cp313-cp313t`, `cp314-cp314t`). The stable ABI does not exist for
  free-threaded builds, so each needs its own version-specific wheel. The
  extension declares `gil_used = false`, and the smoke lanes fail if importing
  gamfit, or numpy, turns the GIL back on.

The list of interpreters comes from `pyproject.toml`:
`python3 scripts/wheel_smoke.py interpreters` prints the abi3 floor and the
free-threaded versions, and the build and smoke matrices both read it. A new
Python version with a free-threaded build gets its wheel and its smoke lane
once its version classifier is added.

Plus the sdist. The sdist carries what building the extension reads and
nothing else: `[tool.maturin] exclude` in `pyproject.toml` drops integration
tests, `cfg(test)` modules, fixtures, examples, helper binaries and
benchmarks, and `scripts/release_footprint.py sdist` compiles the unpacked
sdist with its own `Cargo.lock` and refuses any file that compilation did not
read. It runs on every pull request touching the crates or the packaging
(`release-footprint.yml`) and on the sdist `pypi-wheels.yml` uploads.

## PyPI's storage limit

PyPI keeps every file of every release, yanked ones included, against one
project size limit (10 GiB by default) and a per-file limit (100 MiB by
default). twine uploads file by file, so a project that fills up mid-upload is
left with a release that has some wheels and not others.

Before the first upload, the release job runs
`scripts/release_footprint.py storage dist`. It reads what the project already
holds from PyPI's JSON simple API, adds every file of this release PyPI does
not yet have, and fails the job if that total would pass 90% of the limit or if
any file is over the per-file limit. Nothing is uploaded when it fails.

Where the project stood when the guard was added (read from
`https://pypi.org/simple/gamfit/`):

| | |
|---|---|
| Stored | 8,289,947,778 bytes (7.72 GiB) in 434 files, 69 releases (0.1.174 to 0.1.267) |
| Latest full release (0.1.267) | 245.8 MB: seven abi3 wheels of about 32 MB and a 20 MB sdist |
| A full release with free-threaded wheels | about 0.69 GB: 21 wheels and a 15 MB sdist |
| Room below the 90% threshold | 1.37 GB, about two full releases |

None of the 69 releases is a PEP 440 pre-release, so there are no
pre-releases to prune as such; the space is held by 0.1.x releases that are
superseded within days.

### When the guard refuses a release

These are PyPI account actions; no workflow can take them.

1. **Delete superseded releases.** A project owner removes old 0.1.x releases
   under *Manage project → Releases* on pypi.org. Deletion is permanent, and
   the deleted filenames can never be uploaded again, so keep every release a
   user may have pinned (anything referenced from an issue, a paper or the
   changelog as a known-good version). Releases range from 46 MB (two-file
   releases such as 0.1.260) to 246 MB, so the full eight-file releases free
   the most space each.
2. **Ask PyPI for a larger limit.** Open a project size limit request with
   the issue templates at <https://github.com/pypi/support/issues/new/choose>
   for `gamfit`, giving the per-release size above and the number of platforms
   and Python builds it covers.
3. **Record a granted limit.** PyPI's API does not report a raised limit, so
   once one is granted, set the repository variable
   `PYPI_PROJECT_SIZE_LIMIT_GIB` (*Settings → Secrets and variables → Actions
   → Variables*) to the new limit in GiB. The release job passes it to the
   guard; without it the guard assumes PyPI's default.

## Size decisions

* **The sdist cannot be made smaller than about 15 MB by excluding files.**
  After the excludes it is 14.4 MiB (1,200 files), down from 20.3 MiB
  (1,701 files). What remains is the Rust source the extension compiles,
  about 61 MB uncompressed: gam-models, gam-sae and gam-solve are about a
  fifth of it each. The sdist check proves every one of those files is read
  by the build.
* **The manifold SAE code stays in the one extension.** Moving it into a
  second extension module inside the same wheel would not change what PyPI
  stores. Shipping it as a separate distribution is not possible without
  first untangling the crates: `gam-inference` re-exports `gam-sae`, and
  `gam-models` and `gam-predict` depend on `gam-inference`, so the SAE code is
  on the core fit path's dependency graph. Its share of the compiled source
  (about 21% of the sdist) is also under the 30% saving that would justify the
  split.

## conda-forge

`packaging/conda-forge/meta.yaml` is a draft recipe for a `gamfit` feedstock,
built from the PyPI sdist. It is not submitted: submitting it to
`conda-forge/staged-recipes` is a maintainer action once a release carrying
the sdist excludes is on PyPI and its `sha256` is filled in.

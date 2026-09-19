# release-footprint

TITLE: Bound PyPI storage per release, add free-threaded wheels (release prep only, no publish)
WORK ITEM: Read audit/packaging.md PKG-07 and PKG-09, and audit/docs.md DOC-4 / audit/accuracy.md B1 (wheel lacks the cubature fallback). wheel-ci covers the build matrix, not the storage budget or cp313t.
Evidence: PyPI project storage is 8.29 of 10 GB at about 246 MB per release, about 7 releases of headroom. The .so is 79.5 MB. The sdist includes test data and tests.rs. The module declares gil_used=false, but no cp313t/cp314t wheels are published. There is no conda-forge recipe.
Fix: add `[tool.maturin] exclude` for sdist test data and tests.rs in pyproject.toml. Add a CI step in .github/workflows/pypi-wheels.yml that queries project size and fails above 90% before upload. Document the human actions (prune pre-releases, request a limit increase) in RELEASING or CONTRIBUTING, not a new report. Add cp313t/cp314t to the wheel matrix. Optionally split SAE/manifold into a separate extension if it saves > 30%. Draft a conda-forge recipe (meta.yaml) under packaging/. Do NOT publish a release or upload to PyPI; leave release cutting to the maintainer and note in the PR that the gate is ready.
Coordinate: wheel-ci (same pyproject.toml and workflows; land after it or in one agreed PR), cleanup.
Acceptance: a CI test asserts that the sdist built by `maturin sdist` contains no tests/data or tests.rs and is < 5 MB. A workflow dry-run with a mocked size of 9.5 GB fails the guard. A cp313t wheel imports and runs a fit in CI. All fail at HEAD.

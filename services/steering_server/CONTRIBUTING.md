# Contributing

Keep changes scoped to the steering server package unless a shared repo-level change is necessary.

Quality gates intended for CI:

```bash
ruff check app tests
mypy app
pytest
docker build .
helm lint helm/steering-server
```

Do not add compatibility shims or fallback execution paths. If a production integration is not implemented, mark it as a clear `stub: to-be-implemented` and keep the callable API deterministic.

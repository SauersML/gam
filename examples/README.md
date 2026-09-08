# Runnable examples

Every Python file in this directory is a self-contained example that creates its
own synthetic data. Run one from the repository root, for example:

```console
python examples/streaming_matern_demo.py
```

The examples require an installed `gamfit` extension. A development checkout can
prepare that extension with `uv run maturin develop --release`.

Issue reproductions, performance probes, corpus harvesters, and other scripts
that need operator-provided data or hardware live in [`experiments/`](../experiments/).

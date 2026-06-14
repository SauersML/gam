# Performance Repro Scripts

Standalone profiling scripts that are intentionally outside the main benchmark
runner.

`profile_repro1082.sh` samples the `repro1082_slow_quality` release example
with `eu-stack`, stores raw samples under `/tmp/prof_<case>`, prints the last
run log lines, and aggregates the most frequent stack symbols.

Build the example first, then run:

```bash
cargo build --release --example repro1082_slow_quality
bash bench/perf/profile_repro1082.sh negbin_syn 80 0.25
```

Valid cases are defined by `examples/repro1082_slow_quality.rs`:
`negbin_syn`, `negbin_real`, `cyclic`, `poisson_real`, and `gaussian_te`.

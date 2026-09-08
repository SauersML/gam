# Timewiggle coefficient-map benchmark

Run on MSI with an existing successful rustc argv receipt and warm dependencies:

```sh
python3 bench/measurements/issue_932/run_timewiggle.py \
  --dependencies .buildd/issue932-kernel-direct-argv.json \
  --log .buildd/issue932-timewiggle.log --cpus 88
```

The runner compiles the **actual production scalar map** through a Rust `path`
module, using the receipt's `gam_math` and `ndarray` artifacts. It builds no
dependencies, uses opt-level 3 / one codegen unit / LTO off, and caps compilation
at 45 seconds and testing at 30 seconds. Choose an available CPU on the shared
host. The exact rustc argv is saved alongside the log.

The gate compares the supplied-basis map against the polynomial/exp scalar
program from the July #932 reopening. It checks every output channel over 64
varied rows, then times 15 interleaved repetitions with 64 rows per arm call.
Both `Dual2<Order2<5>>` and `Dual2<OneSeed<5>>` must be strictly faster. The same
test is part of `gam-models` and is discovered by `scripts/speed_gates.py`.

This isolates development iteration. Integrated release CI remains required,
as does the issue's separate strongest-hand runtime-width comparison. The
analytic basis program here uses scalar jet operations; it is not a fully
hand-expanded spline derivative schedule. Correctness additionally compares
against direct polynomial arithmetic at widths 0, 1, 2, 7, and 32, with live
coefficient jets and nonzero mixed channels.

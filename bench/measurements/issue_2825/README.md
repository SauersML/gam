# Streaming block update diagnostic

The recorded run uses the first 512 rows of the cached Qwen activation artifact
from issue #2825's MSI job 18095566. It retains all 2,560 features and uses 256
atoms, block size 2, top-8 blocks, no births, and the default tolerance 1e-6.
The raw activation data are not included. These are training diagnostics on a
small prefix, not held-out quality results or a matched old/new comparison.

Run on MSI acn112 with four Rayon workers, the existing q2 build cache, and
shared-disk storage. For reproduction, replace `WARM_TARGET` and `CORPUS.npy`
with the existing warm target and activation artifact paths, and configure
`TMPDIR` to use shared disk storage before building:

```sh
RAYON_NUM_THREADS=4 cargo build --locked -j4 --profile test \
  --config 'profile.test.package.gam-sae.codegen-units=16' \
  --target-dir WARM_TARGET \
  -p gam-sae --example block_stream_convergence
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  WARM_TARGET/debug/examples/block_stream_convergence \
  CORPUS.npy 512 256 2 8 12
```

EV increases on all 12 recorded passes, from 0.467463 to 0.717002. Both gamma
and frame residuals remain above tolerance, so the diagnostic exits with a
nonconvergence error and produces no fit artifact. The original 100,000-row,
8,192-atom case has not been rerun. `plot.py` renders only the recorded JSON;
it does not fit or rerun a model.

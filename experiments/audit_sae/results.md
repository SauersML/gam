# SAE Audit Surface

Date: 2026-07-06

API shipped in this branch:

```python
gamfit.audit_sae(checkpoint, activations, *, codes=None, decoder_key=None, active=None, ...)
```

Supported external checkpoint format for the facade:

- `.npy` decoder matrix with shape `K x P`, one dictionary row per atom and one activation dimension per column.
- `.npz` / `.safetensors` containing a `decoder` tensor, or an explicit `decoder_key`.
- Python mapping or array with the same `K x P` decoder contract.

The facade is thin over Rust. If `codes` are supplied, Rust audits those frozen external encoder activations. If `codes` are absent, the facade calls the Rust sparse router against the frozen decoder, densifies the returned sparse layout, and calls the Rust audit entry point. The structured report includes the dual certificate, birth candidates, routability floor and empirical dark-matter fraction, per-firing coordinate SEs for harmonic blocks, per-atom Betti topology summaries, and an atlas-nerve report when a block dictionary has selected composable charts.

## MSI verification

Required crate-local gate:

```text
cd <DATA_ROOT>/gam_cx_audit
. $HOME/.config/gam-build-env
export CARGO_TARGET_DIR=<DATA_ROOT>/scratch/target_shared
cargo check -p gam-sae
```

Result: exit code 0. Last lines captured in `/tmp/gam_cx_audit_check.log`:

```text
Some errors have detailed explanations: E0432, E0463.
For more information about an error, try `rustc --explain E0432`.
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.65s
```

The two diagnostics are stale/unrelated `gam-solve` messages emitted before Cargo reported `Finished`; the command exit status was 0.

## Gemma Scope 2 audit

Blocked before producing audit numbers.

What blocked it:

- The available MSI gamfit virtualenvs expose neither `gamfit.audit_sae` nor the Rust `audit_sae` pyfunction.
- Building the updated `gam-pyffi` extension on MSI is the right path, but `cargo check -p gam-pyffi` enters the root `gam` build script and spent more than 25 minutes in `build-script-build` with no log progress. I stopped that extra check to avoid leaving a runaway build. The required `gam-sae` check above was already green.
- Because the updated Python extension was not available, I did not run the Gemma Scope 2 audit and did not fabricate dual-certified atom counts, dark-matter fractions, or Betti distributions.

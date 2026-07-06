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

Required crate-local gate, captured by `experiments/audit_sae/verification_gate.sh`:

```text
cd <DATA_ROOT>/gam_cx_audit
. $HOME/.config/gam-build-env
cargo check -p gam-sae --target-dir <DATA_ROOT>/scratch/target_shared
```

Result: exit code 0. The gate harness captures stdout and stderr separately and refuses to write the `results.md` verdict unless both captured streams are free of Rust compiler diagnostics.

Certified gate stderr excerpt:

```text
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.65s
```

## Gemma Scope 2 audit

Blocked before producing audit numbers.

What blocked it:

- The available MSI gamfit virtualenvs expose neither `gamfit.audit_sae` nor the Rust `audit_sae` pyfunction.
- No extension-build output is part of this verification artifact. The only certified gate here is the crate-local `gam-sae` check captured above.
- Because the updated Python extension was not available, I did not run the Gemma Scope 2 audit and did not fabricate dual-certified atom counts, dark-matter fractions, or Betti distributions.

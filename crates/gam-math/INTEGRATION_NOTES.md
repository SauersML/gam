# gam-math Integration Notes

The moved jet files contain these references outside the moved set:

- `src/jet_tower.rs`: `crate::probability::signed_probit_logcdf_and_mills_ratio`
- `src/jet_tower.rs`: `crate::probability::log1mexp_positive`
- `src/jet_tower.rs`: doc link to `super::row_kernel::RowKernel`
- `src/jet_scalar.rs`: doc link to `super::row_kernel::RowKernel`
- `src/jet_poisson_oracle_tests.rs`: doc link to `super::row_kernel::RowKernel`

They were left unchanged because this extraction is scoped to moving the jet files,
fixing internal jet cross-references, and scaffolding `gam-math` without a `gam`
dependency.

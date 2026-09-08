# Issue 2833 verification

The final MSI extension has SHA256 `7112c1fffe857fb5eaf709c93bbc6de6b899925329f287b53ef66ba506c7509c` and was loaded from `/projects/standard/hsiehph/sauer354/gam-main-validation/gamfit/_rust.abi3.so`. It was built with the existing `gam-deslop-a2-target` dev cache on CPUs 108–109, then extracted from the wheel and installed by atomic rename. The final build completed in 2m52s.

`issue2833-final-extension-source-sha256.json` records the shared source snapshot before compilation. Other workers' existing changes were present. The root build script was unchanged; its existing source-archive policy skipped tracked-file audits because this remote source directory has no `.git`. The remaining build checks ran normally. The spatial worker formatted its three files after compilation; that formatting is not represented as a new compiled source snapshot.

Final public acceptance, using `.venv-python312-issues/bin/python -m pytest` with two threads and a repository-local temporary directory:

```
tests/test_latent_reml_gradient_2833.py
tests/test_flexible_binomial_response_scale_2748.py
7 passed in 21.20s
```

The latent suite checks all three reported fixtures at finite-difference steps `1e-5` and `1e-6`, smoothing stationarity, initialization-invariant adjoints, the full identity-penalty rank, downhill movement, a coefficient frame independent of other rows, and rejection of a corrupted cached penalty rank. The two flexible-binomial cases verify fit, compiled-model consumption, held-out plugin/posterior predictions, and save/load.

Earlier native validation of the same rank/QR/SVD repair passed the identity-penalty rank and inverse oracle at penalty scales `1e-8`, `1`, and `1e8`, the preserved large-strength block REML oracle, and six existing scalar/multiresponse backward and cache finite-difference tests. A final native rerun was prevented by another worker removing its exact cached `gam_terms` artifact. Final cache-validation cleanup is covered by the public corrupted-cache test above.

`issue2833-acceptance-metrics.json` records the preceding `d449d353...` extension's detailed numerical diagnostics: all three gradient cosines exceed `0.99999999999`, relative errors at the original `1e-6` step are at most `4.33e-6`, and all five normalized downhill steps from `1e-2` through `1e-6` lower the reported score. The final extension repeats the two-step gradient and descent assertions.

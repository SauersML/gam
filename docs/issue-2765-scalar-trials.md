Commit `b4bef4dd1` routes dynamic-slope value trials through the scalar row
likelihood. Previously they evaluated the nine-feature gradient and Hessian and
discarded both. The scalar evaluator uses the same feature geometry and domain
checks as the Newton evaluator.

On MSI, the current model test binary passed all nine follow-up-domain tests in
0.37 seconds. These include scalar/Newton value equality on a nonconstant
follow-up slope and the existing domain and derivative checks. The receipt is
`bench/measurements/issue_triage_20260908/survival-follow-up-current.log`.

The first timing probe retained the entire Newton tuple, so it does not measure
the old production caller fairly. Its timing ratio is not a claimed speedup.
The committed probe now consumes only the likelihood from both evaluators;
that timing correction still needs execution.

The original recovery and outer mode-response acceptance runs remain pending.
Issues #2765 and #2767 remain open. No local build or test was run.

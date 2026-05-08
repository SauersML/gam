# Joint Newton trust-region memory note

The joint Newton inner loop now truncates the proposed step to a carried
trust-region radius before evaluating the likelihood.  The logged line

```text
[PIRLS/joint-Newton/trust-region] cycle=... attempt=... accepted=... rho=... radius=a->b
```

records the gain ratio distribution and trust-radius trajectory needed to audit
biobank-scale runs.  `rho` is `(old_objective - trial_objective) /
predicted_reduction`; accepted steps have positive actual decrease and positive
predicted decrease.  Radius updates follow the standard rule used by the inner
loop: shrink by `0.25` for rejected/poor (`rho < 0.25`) steps, double for very
good boundary (`rho > 0.75`) steps, and otherwise carry the radius forward.

For the motivating cycle-0 pattern that previously accepted the fifth
backtracking trial (`alpha = 0.0625`), the equivalent trust-region trajectory
should show one radius-limited trial in normal cases and at most the emergency
shrunken retry when the initial radius is still too aggressive.  This is the
log-derived source of truth for gain-ratio histograms on real FLEX biobank
runs; the ignored criterion benchmark `trust_region_line_search` isolates the
same before/after evaluation-count pattern without requiring the protected
biobank data.

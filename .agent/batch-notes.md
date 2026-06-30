# Batch 1498-1622 — reopen #1561

## #1561 location-scale scale-block over-smoothing (gaulss)
Reopened: closed COMPLETED as a tracker, but the metric is LIVE and GATING-RED on
the reference-quality dashboard (run 28399514029):
- quality_vs_gamlss_gaussian_location_scale.rs:270 METRIC_OFF pearson(logσ,truth)=0.70609 < 0.80
- :554 METRIC_OFF location-scale NLL 3.3163 did not beat constant-σ 3.1809

Owner narrowed cause to the scale block's λ-SELECTION on a near-flat Fisher surface.
Experiments left "not yet run (require a build)". This agent has a build.

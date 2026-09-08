Fresh MSI regression: the current streaming alternation still admits a routed loss increase with six overcomplete rank-two blocks, 96 rows, eight features, top-k three, and births disabled. The deterministic test completed in 0.026 seconds and failed at epoch 22:

```text
previous explained variance = 0.89490867249443684
current explained variance  = 0.89391699006944969
```

This uses the complete current working source, including the repaired new-gamma frame step and f64 transport. It is not a rerun of the historical executable. No convergence or issue resolution is claimed.

The remaining mechanism is the distinction between fixed supports and rerouted supports. The polar surrogate decreases the tied reconstruction loss with the pass's support frozen; the next pass runs top-k selection again. The surrogate does not bound that changed objective.

The repair under development treats frame proposals as paired full-pass trials. Candidate and baseline are rerouted on the same streamed rows, and each receives its own analytic gamma profile. A rejected frame proposal takes a shorter retracted step along the same direction. Rejection never certifies stationarity, and the streaming state retains only bounded model/moment storage. The new regression remains strict about measurable EV improvement and monotonicity. Verification of this repair is pending.

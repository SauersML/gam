The accumulator's two terms must use one probability measure. The existing
axis nodes at one standard deviation with mass 1/(2r) integrate only V/r,
while summing coefficient chords substitutes V in the other term. No positive
probability measure can simultaneously represent those two choices.

For asymmetric standardized axis distances a_j and b_j, matching zero mean
and unit second moment requires masses 1/[a_j(a_j+b_j)] and
1/[b_j(a_j+b_j)]. Their combined mass is 1/(a_j b_j). At the present
one-standard-deviation nodes, summing these masses gives r, so a positive
center weight cannot repair the rule when r exceeds one.

The local implementation now uses calibrated rays only to construct a Gaussian
proposal. It evaluates that proposal's spherical nodes at radius sqrt(r),
weights them by the actual criterion divided by the proposal density, and
normalizes the positive weights. The proposal density is equal at these nodes,
so it cancels during normalization. Conditional covariance and centered
coefficient covariance use exactly the same nodes and weights. Untreated
directions retain their independent linear Gaussian response.

Seven new analytical tests cover varying quadratic conditional covariance plus
linear Gaussian means, asymmetric weights, variation between axis-pair means,
splitting mass without changing the measure, translation/response scaling,
independent residual directions, and refusal of signed or empty mass. The
quadratic conditional-covariance fixture directly observes the missing-rank
defect that constant-covariance fixtures cannot detect.

This is not yet a verified resolution of #1561. The new tests and empirical
interval calibration still require MSI execution. The transport dropped twice
before the new remote build started. No PSD clipping or substitution of a
first-order covariance has been added, and marginal covariance being wider than
the mode-conditional covariance is treated as an empirical calibration check,
not an algebraic consequence of total covariance.

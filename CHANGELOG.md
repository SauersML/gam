## Unreleased

- **A constrained-Laplace VALUE refused on a moment-matching condition it never uses, at the sites
  EP had not yet moved** (gam#4571). `share_and_magnitude` reads one number from a site, the tilted
  log mass, and obtained it by calling `site_update`, which computes the full cumulant jet and
  refuses on a non-positive tilted variance before returning. The variance enters only through
  `τ̃ = 1/κ₂ − τ_c`, which the share never forms. The share is also evaluated once at sweep 0, at
  the boundary-limit starting sites, where the module's own premise is that the start "only selects
  where EP begins" and its fixed point does not depend on it. So a fit could be refused for a
  property of a starting point the algorithm states is arbitrary, before a single sweep. The share
  now reads the mass through `tilted_log_mass`, which refuses only where the half-line has no
  decay, and the fold verdict stays where the moment matching is.
- **The fold refusal now names the cavity that produced it** (gam#4571). The verdict is a sign test
  on a cumulant that vanishes at the continuation's own boundary `τ_c = −ν_c²/2`, so a mode near
  that boundary returns a small negative number and the number alone cannot say whether it is past
  the fold or unresolved from zero. The refusal carries the cavity precision, the cavity shift and
  that boundary, together with the two operands the cavity is the difference of, `tau_i` and
  `Sigma_ii`, and their product printed to full precision: by the Sherman-Morrison identity
  `1/Sigma_ii - tau_i = 1/(a_i^T B_i^-1 a_i)` exactly, so how far that product sits from 1 is the
  whole cavity, and a product at `1 +/- 1e-13` says the cavity is rounding rather than curvature.
  It also carries `M`'s curvature along the offending row's normal, filled where the precision and
  the caller's rows are both in scope, as a field rather than a log line, since it is evidence the
  verdict needs and not a diagnostic beside it. That evidence is one-sided by construction: a
  negative value settles that the leave-one-out precision is not positive definite, while a
  non-negative one settles nothing, because that matrix needs the precision positive in every
  direction coupling to the row's normal and not only along it. Its absence is a statement too: a
  fold raised on the derivative path carries no curvature, because there the sites are at their
  converged fixed point, the dominance the identity needs fails, and the cavity precision is what
  to read instead. The refusal also prints the product's distance from 1 against a band taken from
  the Cholesky's own pivot extremes, `(max/min)^2` being a free lower bound on the condition number
  from numbers the factorization has already produced. Being a lower bound makes that test one-
  sided too: inside the band the cavity's sign is the subtraction's for certain, while outside it
  nothing is established. Its pin asserts
  the reported cavity rather than matching on `..`. A negative control is added
  beside it: the same one-row geometry with an indefinite precision INSIDE the continuation must
  price, since a refusal that fired on every indefinite precision would be indistinguishable from
  the fold one, and an indefinite precision inside the regime is what the continuation exists for.

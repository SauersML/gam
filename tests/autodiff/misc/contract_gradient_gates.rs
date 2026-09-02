

// #1255 noise-matched bar for the non-canonical (probit) GLM-REML outer-gradient
// row. The binomial/probit inner PIRLS solve leaves an O(1e-7..1e-8) noise floor
// on V(ρ); a tight 5e-6 central step over that floor is dominated by noise. A
// wider central step (the standard √(round-off)/curvature trade-off) lifts the FD
// signal cleanly above the floor, and the bar is set to match the residual
// truncation+noise budget. This is NOT a weakened exact-arithmetic bar — a wrong
// analytic gradient disagrees by O(1) relative, orders of magnitude above this.


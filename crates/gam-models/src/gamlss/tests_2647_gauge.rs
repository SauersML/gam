// gam#2647 — the joint penalized Hessian of the binomial location-scale wiggle
// model must be non-singular, and that has to be the PENALTY's doing.
//
// The measurement instruments that found this (a budget ladder, a design-level
// alias ladder, an orbit walk) are recorded on the issue; what survives here is
// the property they established, asserted rather than printed.
#![cfg(test)]


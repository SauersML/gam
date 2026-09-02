//! #1939 OBJECTIVE-QUALITY acceptance bar — existence/intensity DECOUPLING in the
//! physical dictionary. The current representation carries intensity directly in
//! each fitted decoder. *Existence* (does this atom explain held-out structure)
//! must be identified separately from *intensity* (how large its contribution is).
//!
//! We plant that ground truth: two live circles on DISJOINT output subspaces whose
//! amplitudes differ by ~an order of magnitude, plus a DEAD atom slot with no
//! planted signal, and fit a K=3 dictionary. The objective is truth recovery —
//! the planted amplitude ratio and the dead/alive partition — NOT reproduction of
//! any reference tool's fitted parameters.
//!
//! Intensity lives in the decoder magnitude `‖B_k‖`; the executable objective bar
//! below therefore identifies a dead atom by its held-out contribution while
//! separately checking the recovered intensity ratio of the two live atoms.


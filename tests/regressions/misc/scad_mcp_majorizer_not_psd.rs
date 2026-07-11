//! Bug hunt: `ScadMcpPenalty` does not supply a PSD majorizer.
//!
//! The `AnalyticPenalty` trait documents `psd_majorizer_diag` as the diagonal
//! of a **PSD majorizer** `B` of the Hessian, with the contract that
//! `B ⪰ ∂²P/∂target²` everywhere AND `B ⪰ 0`. Nonconvex penalties (whose exact
//! Hessian is indefinite) are required to override the default so that the
//! inner Newton / PIRLS solve and the log-det / preconditioner pipeline get a
//! genuinely positive-semidefinite curvature block.
//!
//! SCAD and MCP are nonconvex by construction: their exact diagonal Hessian
//! `hess_one` is `weight·eps²/r³ − 1/γ` (MCP) and goes to `≈ −1/γ` < 0 across
//! the whole taper/active region. But `ScadMcpPenalty` never overrides
//! `psd_majorizer_diag`, so it inherits the trait default which simply returns
//! `hessian_diag` — the exact, *negative* Hessian. The resulting "majorizer" is
//! therefore not PSD, violating `B ⪰ 0`.
//!
//! This is the same failure mode as the (closed) smooth-threshold majorizer bug,
//! but a distinct penalty: `SmoothThresholdPenalty` at least overrides the method; `ScadMcpPenalty`
//! does not override it at all.
//!
//! This test must FAIL on the current code (majorizer == exact negative
//! Hessian) and PASS once `ScadMcpPenalty` supplies a proper PSD majorizer
//! (e.g. the positive reweighted-ℓ² curvature `weight·eps²/r³`, dropping the
//! concave `−1/γ` term). No edits to this test should be needed for it to pass.

use gam::terms::analytic_penalties::{AnalyticPenalty, PenaltyConcavity, PsiSlice, ScadMcpPenalty};
use ndarray::{Array1, array};

#[test]
fn scad_mcp_psd_majorizer_is_actually_psd() {
    // Targets that land in the taper / active region where the exact nonconvex
    // Hessian is negative for both variants. (For SCAD the first region |t| <= w
    // stays convex; the larger |t| entries exercise its concave middle region.)
    let probe = array![0.02_f64, 0.4, 0.7, 0.9, -1.1, -0.05];
    let rho = Array1::<f64>::zeros(0);

    for variant in [PenaltyConcavity::Mcp, PenaltyConcavity::Scad] {
        let n_eff = probe.len();
        let target = PsiSlice::full(n_eff, Some(1));
        let gamma = match variant {
            PenaltyConcavity::Mcp => 3.0,
            PenaltyConcavity::Scad => 3.7,
        };
        let pen = ScadMcpPenalty::new(target, 0.5, n_eff, gamma, 1.0e-4, variant, false)
            .expect("valid ScadMcpPenalty");

        let maj = pen
            .psd_majorizer_diag(probe.view(), rho.view())
            .expect("ScadMcp is coordinate-separable: a diagonal majorizer exists");
        let hess = pen
            .hessian_diag(probe.view(), rho.view())
            .expect("ScadMcp exposes an analytic diagonal Hessian");

        for i in 0..probe.len() {
            // Contract part 1: B ⪰ 0.
            assert!(
                maj[i] >= -1e-12,
                "{variant:?}: psd_majorizer_diag[{i}] = {} is negative at t = {}, \
                 violating the documented B ⪰ 0 contract",
                maj[i],
                probe[i]
            );
            // Contract part 2: B ⪰ ∂²P (a majorizer dominates the exact Hessian).
            assert!(
                maj[i] >= hess[i] - 1e-12,
                "{variant:?}: psd_majorizer_diag[{i}] = {} is below the exact Hessian {}, \
                 violating the documented B ⪰ ∂²P contract",
                maj[i],
                hess[i]
            );
        }
    }
}

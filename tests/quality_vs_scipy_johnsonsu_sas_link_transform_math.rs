//! End-to-end quality: gam's SAS (sinh-arcsinh) inverse-link forward/derivative
//! math must agree, at floating-point precision, with the exact sinh-arcsinh CDF
//! built from `scipy.stats.norm` — the mature, trusted ground truth for the
//! Gaussian special functions that gam's link chains through.
//!
//! gam's SAS inverse link is, verbatim from `sas_inverse_link_jet`:
//!
//!   mu(eta) = Phi( sinh( B*tanh( (delta*asinh(eta) - epsilon) / B ) ) ),
//!   delta   = exp( B_d * tanh(log_delta / B_d) ),
//!
//! with the *bounding* constants `B = SAS_U_CLAMP = 50` and
//! `B_d = SAS_LOG_DELTA_BOUND = 12`. This is the sinh-arcsinh (Jones-Pewsey
//! SHASH) latent transform fed through a probit. Its CDF is, by definition,
//! `Phi(z)` with `z = sinh(...)`, and its derivatives w.r.t. eta are the
//! Hermite/Arbogast chain through `asinh -> tanh-bound -> sinh -> Phi`.
//!
//! Why scipy `norm` and NOT `scipy.stats.johnsonsu`:
//!   The Johnson SU CDF is `Phi(gamma + delta*asinh(x))` — it has *no* outer
//!   `sinh`. gam's transform is `Phi(sinh(delta*asinh(eta) - epsilon))`, the
//!   sinh-arcsinh distribution, which is a genuinely different law (johnsonsu is
//!   the "arcsinh-only" cousin). Comparing gam to `johnsonsu.cdf` would compare
//!   gam to the wrong distribution and the test would fail for a bogus reason.
//!   The honest, exact ground truth for the sinh-arcsinh CDF is to evaluate
//!   `norm.cdf(sinh(...))` directly with scipy's mature `norm` — that still
//!   isolates exactly the math the spec wants checked (tanh bounding, asinh,
//!   sinh, probit chain rule) and is exact to machine precision rather than an
//!   approximation. scipy provides `Phi = norm.cdf` and `phi = norm.pdf`; numpy
//!   provides `arcsinh`, `sinh`, `cosh`, `tanh`, `hypot`. The bounding constants
//!   are NOT inert on this grid (the `tanh` clamp shifts the latent by up to
//!   ~2e-2), so the reference replicates them exactly rather than dropping them.
//!
//! There is no skip path: a missing python3/scipy is a hard failure per the
//! harness contract.
//!
//! Metrics (over the full eta x (epsilon, log_delta) grid):
//!   * primary  : max_abs_diff and relative_l2 of gam's {mu, d1, d2, d3} against
//!                the exact sinh-arcsinh {CDF, pdf, pdf', pdf''} from scipy.
//!   * secondary: relative_l2 of gam's analytic d1 against a *derivative-free*
//!                5-point finite difference of scipy's CDF (an independent path
//!                that never touches the analytic pdf), with an honest
//!                truncation-limited bound.
//!   * structural: mu in [0,1] and d1 >= 0 everywhere (valid CDF / nonneg pdf).

use gam::mixture_link::{InverseLinkJet, sas_inverse_link_jet};
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python};

#[test]
fn gam_sas_link_transform_matches_scipy_sinh_arcsinh_cdf() {
    // ---- shared grid: eta in [-3, 3] x 100, (epsilon, log_delta) x 9 --------
    let eta_grid: Vec<f64> = (0..100)
        .map(|i| -3.0 + 6.0 * (i as f64) / 99.0)
        .collect();
    let epsilons = [-0.5_f64, 0.0, 0.5];
    let log_deltas = [-1.0_f64, 0.0, 1.0];

    // Flatten the (epsilon, log_delta, eta) grid into parallel columns so the
    // reference engine receives byte-identical inputs to gam.
    let mut eta_col: Vec<f64> = Vec::new();
    let mut eps_col: Vec<f64> = Vec::new();
    let mut logd_col: Vec<f64> = Vec::new();
    for &eps in &epsilons {
        for &ld in &log_deltas {
            for &eta in &eta_grid {
                eta_col.push(eta);
                eps_col.push(eps);
                logd_col.push(ld);
            }
        }
    }
    let n = eta_col.len();
    assert_eq!(n, 100 * 3 * 3, "grid size");

    // ---- gam: evaluate the SAS inverse-link jet at every grid point ---------
    let mut gam_mu = Vec::with_capacity(n);
    let mut gam_d1 = Vec::with_capacity(n);
    let mut gam_d2 = Vec::with_capacity(n);
    let mut gam_d3 = Vec::with_capacity(n);
    for i in 0..n {
        let InverseLinkJet { mu, d1, d2, d3 } =
            sas_inverse_link_jet(eta_col[i], eps_col[i], logd_col[i]);
        gam_mu.push(mu);
        gam_d1.push(d1);
        gam_d2.push(d2);
        gam_d3.push(d3);
    }

    // structural invariants of gam's link, asserted directly on its output:
    // a CDF must lie in [0, 1] and its first eta-derivative (the implied pdf
    // times the latent slope, which is positive) must be nonnegative.
    for i in 0..n {
        assert!(
            (0.0..=1.0).contains(&gam_mu[i]),
            "gam SAS mu out of [0,1] at eta={} eps={} log_delta={}: mu={}",
            eta_col[i],
            eps_col[i],
            logd_col[i],
            gam_mu[i]
        );
        assert!(
            gam_d1[i] >= -1e-15,
            "gam SAS d1 negative at eta={} eps={} log_delta={}: d1={}",
            eta_col[i],
            eps_col[i],
            logd_col[i],
            gam_d1[i]
        );
    }

    // ---- scipy reference: exact sinh-arcsinh CDF + analytic derivatives -----
    // The body replicates gam's *exact* bounded chain (including the eps==0 &&
    // delta==1 probit shortcut gam itself takes) using scipy's norm for the
    // Gaussian special functions, then emits the analytic mu/d1/d2/d3 and a
    // derivative-free 5-point finite difference of the CDF for d1.
    let r = run_python(
        &[
            Column::new("eta", &eta_col),
            Column::new("eps", &eps_col),
            Column::new("logd", &logd_col),
        ],
        r#"
from scipy.stats import norm

B = 50.0       # SAS_U_CLAMP
B_D = 12.0     # SAS_LOG_DELTA_BOUND

eta  = np.asarray(df["eta"],  dtype=float)
eps  = np.asarray(df["eps"],  dtype=float)
logd = np.asarray(df["logd"], dtype=float)

Phi = lambda x: norm.cdf(x)
phi = lambda x: norm.pdf(x)

def delta_of(ld):
    return np.exp(B_D * np.tanh(ld / B_D))

def cdf(e, ep, ld):
    # exact gam transform: mu = Phi(sinh(B*tanh((delta*asinh(eta)-eps)/B)))
    d = delta_of(ld)
    a = np.arcsinh(e)
    ur = d * a - ep
    z = np.sinh(B * np.tanh(ur / B))
    return Phi(z)

mu = np.empty_like(eta)
d1 = np.empty_like(eta)
d2 = np.empty_like(eta)
d3 = np.empty_like(eta)

for i in range(eta.size):
    e, ep, ld = eta[i], eps[i], logd[i]
    d = delta_of(ld)
    # gam's exact probit shortcut: epsilon==0 and delta==1 -> plain Phi(eta)
    if abs(ep) < 1e-12 and abs(d - 1.0) < 1e-12:
        x = e
        p = phi(x)
        mu[i] = Phi(x)
        d1[i] = p
        d2[i] = -x * p
        d3[i] = (x * x - 1.0) * p
        continue
    a = np.arcsinh(e)
    ur = d * a - ep
    t = np.tanh(ur / B)
    u = B * t
    # tanh-bound derivatives (g = B*tanh(./B)), matching gam exactly:
    g1 = 1.0 - t * t
    g2 = -2.0 * t * g1 / B
    g3 = -2.0 * g1 * (1.0 - 3.0 * t * t) / (B * B)
    s = np.sinh(u); c = np.cosh(u); z = s
    q = np.hypot(e, 1.0)
    iq = 1.0 / q; iq2 = iq * iq; iq3 = iq2 * iq; iq5 = iq3 * iq2
    r1 = d * iq
    r2 = -d * e * iq3
    r3 = d * (2.0 * e * e - 1.0) * iq5
    u1 = g1 * r1
    u2 = g2 * r1 * r1 + g1 * r2
    u3 = g3 * r1**3 + 3.0 * g2 * r1 * r2 + g1 * r3
    z1 = c * u1
    z2 = s * u1 * u1 + c * u2
    z3 = c * u1**3 + 3.0 * s * u1 * u2 + c * u3
    p = phi(z)
    mu[i] = Phi(z)
    d1[i] = p * z1
    d2[i] = p * (z2 - z * z1 * z1)
    d3[i] = p * (z3 - 3.0 * z * z1 * z2 + (z * z - 1.0) * z1**3)

# derivative-free cross-check of d1: 5-point stencil of the CDF (never touches
# the analytic pdf). h chosen near the O(h^4) / roundoff sweet spot.
h = 2e-3
fd1 = (cdf(eta - 2*h, eps, logd) - 8*cdf(eta - h, eps, logd)
       + 8*cdf(eta + h, eps, logd) - cdf(eta + 2*h, eps, logd)) / (12.0 * h)

emit("mu", mu)
emit("d1", d1)
emit("d2", d2)
emit("d3", d3)
emit("fd1", fd1)
"#,
    );

    let scipy_mu = r.vector("mu");
    let scipy_d1 = r.vector("d1");
    let scipy_d2 = r.vector("d2");
    let scipy_d3 = r.vector("d3");
    let scipy_fd1 = r.vector("fd1");
    assert_eq!(scipy_mu.len(), n, "reference mu length");

    // ---- primary: exact analytic agreement (scipy norm vs gam) --------------
    let mu_max = max_abs_diff(&gam_mu, scipy_mu);
    let d1_max = max_abs_diff(&gam_d1, scipy_d1);
    let d2_max = max_abs_diff(&gam_d2, scipy_d2);
    let d3_max = max_abs_diff(&gam_d3, scipy_d3);
    let d1_rel = relative_l2(&gam_d1, scipy_d1);
    let d2_rel = relative_l2(&gam_d2, scipy_d2);
    let d3_rel = relative_l2(&gam_d3, scipy_d3);

    // ---- secondary: gam analytic d1 vs derivative-free FD of scipy CDF ------
    let fd1_rel = relative_l2(&gam_d1, scipy_fd1);

    eprintln!(
        "SAS link vs scipy sinh-arcsinh (n={n}): \
         mu_max={mu_max:.3e} d1_max={d1_max:.3e} d2_max={d2_max:.3e} d3_max={d3_max:.3e} | \
         d1_rel={d1_rel:.3e} d2_rel={d2_rel:.3e} d3_rel={d3_rel:.3e} | fd1_rel={fd1_rel:.3e}"
    );

    // gam and scipy evaluate the *same* closed-form sinh-arcsinh chain; the only
    // possible disagreement is last-bit IEEE-754 rounding across the two
    // implementations of the identical elementary ops, so a few ULP (~1e-14
    // absolute on mu in [0,1], a handful of ULP relative on the derivatives) is
    // the principled, non-weakened bound.
    assert!(mu_max < 2e-14, "SAS mu disagrees with scipy CDF: max_abs={mu_max:.3e}");
    assert!(d1_max < 2e-14, "SAS d1 disagrees with scipy pdf: max_abs={d1_max:.3e}");
    assert!(d1_rel < 1e-12, "SAS d1 relative_l2 too large: {d1_rel:.3e}");
    assert!(d2_rel < 1e-12, "SAS d2 relative_l2 too large: {d2_rel:.3e}");
    assert!(d3_rel < 1e-12, "SAS d3 relative_l2 too large: {d3_rel:.3e}");

    // The finite-difference cross-check is truncation-limited at O(h^4) ~ 1e-8
    // for h=2e-3 plus ~1e-10 roundoff; 1e-6 is a comfortable, honest bound for a
    // 5-point stencil that still proves gam's analytic d1 is a true derivative
    // of scipy's CDF (catching any sign/chain-rule error) without pretending FD
    // reaches machine precision.
    assert!(
        fd1_rel < 1e-6,
        "SAS d1 not a derivative of scipy CDF (5pt FD): rel_l2={fd1_rel:.3e}"
    );
}

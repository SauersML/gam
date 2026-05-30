//! End-to-end quality: gam's simplex primitive must reproduce the Aitchison
//! geometry that the mature compositional-data tools (`compositions` /
//! `robCompositions`) implement, on identical data.
//!
//! Mature comparator: R `compositions` (the canonical Aitchison-geometry
//! package; `compositions::acomp` + `mean.acomp` is the closed geometric-mean
//! center, and `compositions::clr` is the centered-log-ratio map) cross-checked
//! against `robCompositions::cenLR` (the CLR used by `robCompositions::gmc`).
//! These two packages are the *only* mature R tools for Aitchison geometry, and
//! the field is otherwise fragmented single-purpose code — that fragmentation is
//! itself the finding here.
//!
//! What gam exposes publicly on the simplex is one primitive:
//! `gam::geometry::simplex::simplex_frechet_mean` — the closure operator
//! (perturbation/scale normalization onto the simplex) composed with the
//! CLR/Aitchison Fréchet mean (component-wise mean of logs, softmax-closed).
//! That is *exactly* the closed geometric-mean center `mean.acomp`. gam does not
//! expose a public Aitchison-distance function, so the spec's pairwise-distance
//! / triangle-inequality checks cannot be run against a gam distance API
//! directly (documented gap). Instead we benchmark the primitive gam *does*
//! expose and assert the intrinsic metric-defining group laws of Aitchison
//! geometry — closure (scale) invariance and perturbation equivariance — which
//! a correct Aitchison barycenter (and hence a correct CLR-Euclidean distance
//! built on the same closure) must satisfy.
//!
//! Assertions:
//!   1. gam's Fréchet mean == `compositions::mean.acomp` element-wise (the
//!      head-to-head: same closed geometric-mean center).
//!   2. gam's CLR-of-the-mean (= log mean minus its average) == the CLR center
//!      that `compositions::clr` / `robCompositions::cenLR` produce, confirming
//!      gam's closure is the genuine CLR-Euclidean (Aitchison) representation.
//!   3. Closure / scale invariance: feeding row-scaled (unclosed) counts gives a
//!      bit-identical mean — the defining invariance behind d(x,y)=d(cx,cy).
//!   4. Perturbation equivariance: mean(x_i ⊕ p) == mean(x_i) ⊕ p, the group law
//!      that makes the Aitchison metric translation-invariant and geodesic.
//!   5. Simplex closure of the output (positive, sums to 1).

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, max_abs_diff, run_r};
use ndarray::Array2;

/// Closed geometric mean reference, built independently in Rust so we can apply
/// the Aitchison perturbation/equivariance laws to gam's *own* output without
/// reaching for a private gam helper.
fn closed_geometric_mean(rows: &[[f64; 3]]) -> [f64; 3] {
    let n = rows.len() as f64;
    let mut log_mean = [0.0_f64; 3];
    for r in rows {
        // closure first (Aitchison treats the row up to scale)
        let total: f64 = r.iter().sum();
        for k in 0..3 {
            log_mean[k] += (r[k] / total).ln() / n;
        }
    }
    let mx = log_mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out = [0.0_f64; 3];
    let mut s = 0.0;
    for k in 0..3 {
        out[k] = (log_mean[k] - mx).exp();
        s += out[k];
    }
    for v in out.iter_mut() {
        *v /= s;
    }
    out
}

/// CLR of a single closed composition: ln(x) - mean(ln(x)).
fn clr(x: &[f64; 3]) -> [f64; 3] {
    let log: [f64; 3] = [x[0].ln(), x[1].ln(), x[2].ln()];
    let m = (log[0] + log[1] + log[2]) / 3.0;
    [log[0] - m, log[1] - m, log[2] - m]
}

/// Perturbation x ⊕ p = closure(x .* p) — the Aitchison "translation".
fn perturb(x: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
    let mut prod = [x[0] * p[0], x[1] * p[1], x[2] * p[2]];
    let s: f64 = prod.iter().sum();
    for v in prod.iter_mut() {
        *v /= s;
    }
    prod
}

/// Deterministic synthetic 3-component compositions. We draw Gamma(shape,1)
/// variates via a fixed-seed LCG + Marsaglia–Tsang and close them, which is the
/// standard route to Dirichlet(shape,shape,shape) samples. shape=1 => uniform on
/// the simplex (low concentration); shape=0.5 => high-concentration corners.
/// The *raw, unclosed* gamma draws are what we hand to both engines, so gam's
/// closure and `compositions`/`robCompositions`'s closure see byte-identical
/// inputs.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        // 53-bit uniform in (0,1)
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v.clamp(f64::MIN_POSITIVE, 1.0 - 1e-15)
    }
    fn normal(&mut self) -> f64 {
        // Box–Muller
        let u1 = self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn gamma(&mut self, shape: f64) -> f64 {
        // Marsaglia–Tsang; boost for shape<1 via x = G(shape+1) * U^(1/shape).
        if shape < 1.0 {
            let g = self.gamma(shape + 1.0);
            let u = self.unit();
            return g * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return (d * v).max(f64::MIN_POSITIVE);
            }
        }
    }
}

#[test]
fn gam_simplex_frechet_mean_matches_compositions_aitchison_center() {
    // ---- fixed-seed synthetic raw gamma draws (two concentrations) ----------
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let n_pairs = 50usize; // 50 pairs => 100 compositions, per spec
    let n = 2 * n_pairs;
    let mut raw: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        // first half low-concentration (shape 1), second half high (shape 0.5)
        let shape = if i < n / 2 { 1.0 } else { 0.5 };
        raw.push([rng.gamma(shape), rng.gamma(shape), rng.gamma(shape)]);
    }

    // Flat columns shared verbatim with R.
    let c1: Vec<f64> = raw.iter().map(|r| r[0]).collect();
    let c2: Vec<f64> = raw.iter().map(|r| r[1]).collect();
    let c3: Vec<f64> = raw.iter().map(|r| r[2]).collect();

    // ---- gam: closure + Aitchison Fréchet mean ------------------------------
    let mut pts = Array2::<f64>::zeros((n, 3));
    for (i, r) in raw.iter().enumerate() {
        for k in 0..3 {
            pts[[i, k]] = r[k];
        }
    }
    let gam_mean_vec = simplex_frechet_mean(pts.view(), None).expect("gam simplex Fréchet mean");
    assert_eq!(gam_mean_vec.len(), 3);
    let gam_mean = [gam_mean_vec[0], gam_mean_vec[1], gam_mean_vec[2]];

    // ---- compositions / robCompositions reference ---------------------------
    let r = run_r(
        &[
            Column::new("c1", &c1),
            Column::new("c2", &c2),
            Column::new("c3", &c3),
        ],
        r#"
        suppressPackageStartupMessages(library(compositions))
        suppressPackageStartupMessages(library(robCompositions))
        X <- as.matrix(df[, c("c1","c2","c3")])
        ac <- acomp(X)                       # closed compositions (Aitchison)
        ctr <- mean(ac)                      # closed geometric-mean center
        ctr <- as.numeric(ctr / sum(ctr))    # ensure unit closure for comparison
        emit("center", ctr)
        # CLR center two independent ways: compositions::clr and robCompositions::cenLR
        clr_comp <- as.numeric(clr(acomp(matrix(ctr, nrow = 1))))
        cl <- robCompositions::cenLR(matrix(ctr, nrow = 1))
        clr_rob  <- as.numeric(if (!is.null(cl$x.clr)) cl$x.clr else as.matrix(cl))
        emit("clr_comp", clr_comp)
        emit("clr_rob", clr_rob)
        "#,
    );
    let ref_center = r.vector("center");
    let ref_clr_comp = r.vector("clr_comp");
    let ref_clr_rob = r.vector("clr_rob");
    assert_eq!(ref_center.len(), 3, "reference center must be 3-component");

    // ===== Assertion 1: head-to-head closed geometric-mean center ============
    let center_err = max_abs_diff(&gam_mean, ref_center);
    // gam and compositions compute the identical closed geometric mean from the
    // same closure; the only differences are float reduction order, so machine
    // precision is the principled bound (1e-10), not a loose tolerance.
    eprintln!(
        "Aitchison center: gam=[{:.6},{:.6},{:.6}] compositions=[{:.6},{:.6},{:.6}] max_abs={:.3e}",
        gam_mean[0],
        gam_mean[1],
        gam_mean[2],
        ref_center[0],
        ref_center[1],
        ref_center[2],
        center_err
    );
    assert!(
        center_err < 1e-10,
        "gam Fréchet mean diverges from compositions::mean.acomp: max_abs={center_err:.3e}"
    );

    // ===== Assertion 2: gam's closure IS the CLR-Euclidean representation =====
    // Distances in this geometry are Euclidean in CLR; confirm gam's mean lands
    // at the same CLR center compositions::clr and robCompositions::cenLR do.
    let gam_clr = clr(&gam_mean);
    let clr_err_comp = max_abs_diff(&gam_clr, ref_clr_comp);
    let clr_err_rob = max_abs_diff(&gam_clr, ref_clr_rob);
    eprintln!(
        "CLR(center): gam=[{:.6},{:.6},{:.6}] clr_err_comp={:.3e} clr_err_rob={:.3e}",
        gam_clr[0], gam_clr[1], gam_clr[2], clr_err_comp, clr_err_rob
    );
    assert!(
        clr_err_comp < 1e-9 && clr_err_rob < 1e-9,
        "gam closure is not the CLR-Euclidean (Aitchison) coords: comp={clr_err_comp:.3e} rob={clr_err_rob:.3e}"
    );

    // ===== Assertion 3: closure / scale invariance (d(x,y)=d(cx,cy)) =========
    // Multiply each composition by an arbitrary positive scalar (different per
    // row) and re-run gam. The closure must make the Fréchet mean identical — the
    // exact invariance that makes the Aitchison distance scale-free.
    let mut rng_s = Lcg(0xC0FF_EE00_D15E_A5E5);
    let mut scaled = Array2::<f64>::zeros((n, 3));
    for (i, rrow) in raw.iter().enumerate() {
        let c = 0.1 + 100.0 * rng_s.unit(); // strictly positive per-row scale
        for k in 0..3 {
            scaled[[i, k]] = rrow[k] * c;
        }
    }
    let scaled_mean = simplex_frechet_mean(scaled.view(), None).expect("gam mean on scaled rows");
    let scaled_arr = [scaled_mean[0], scaled_mean[1], scaled_mean[2]];
    let scale_err = max_abs_diff(&gam_mean, &scaled_arr);
    eprintln!("closure scale invariance: max_abs={scale_err:.3e}");
    assert!(
        scale_err < 1e-12,
        "gam closure is not scale-invariant (Aitchison distance would not be scale-free): {scale_err:.3e}"
    );

    // ===== Assertion 4: perturbation equivariance (group law) ================
    // mean(x_i ⊕ p) must equal mean(x_i) ⊕ p. This is the translation invariance
    // of Aitchison geometry — the property that makes its distance geodesic and
    // makes the triangle inequality hold by isometry to Euclidean CLR space.
    let p = [2.0, 5.0, 0.25];
    let mut perturbed = Array2::<f64>::zeros((n, 3));
    for (i, rrow) in raw.iter().enumerate() {
        // close the raw row, then perturb; gam will re-close internally
        let total: f64 = rrow.iter().sum();
        let closed = [rrow[0] / total, rrow[1] / total, rrow[2] / total];
        let pr = perturb(&closed, &p);
        for k in 0..3 {
            perturbed[[i, k]] = pr[k];
        }
    }
    let perturbed_mean =
        simplex_frechet_mean(perturbed.view(), None).expect("gam mean on perturbed rows");
    let perturbed_arr = [perturbed_mean[0], perturbed_mean[1], perturbed_mean[2]];
    let expected_perturbed = perturb(&gam_mean, &p);
    let equivar_err = max_abs_diff(&perturbed_arr, &expected_perturbed);
    // Cross-check our Rust reference center matches gam too (guards the helper).
    let ref_helper = closed_geometric_mean(&raw);
    let helper_err = max_abs_diff(&gam_mean, &ref_helper);
    eprintln!(
        "perturbation equivariance: max_abs={equivar_err:.3e} (rust-helper agreement={helper_err:.3e})"
    );
    assert!(
        helper_err < 1e-10,
        "independent Rust closed-geometric-mean disagrees with gam: {helper_err:.3e}"
    );
    assert!(
        equivar_err < 1e-12,
        "gam Fréchet mean violates Aitchison perturbation equivariance (metric would not be geodesic): {equivar_err:.3e}"
    );

    // ===== Assertion 5: output lies on the closed simplex ====================
    let sum: f64 = gam_mean.iter().sum();
    eprintln!(
        "simplex closure: sum={sum:.15} min={:.3e}",
        gam_mean.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    assert!(
        (sum - 1.0).abs() < 1e-12 && gam_mean.iter().all(|&v| v > 0.0),
        "gam mean is not a valid closed composition: sum={sum}"
    );
}

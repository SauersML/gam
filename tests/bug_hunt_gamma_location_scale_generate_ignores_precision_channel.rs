//! Regression for #1125, end-to-end through the `gam` binary: `gam generate`
//! for a dispersion location-scale model (`--predict-noise` on Gamma) must
//! reproduce the fitted per-row precision surface `exp(eta_d(x))`, so synthetic
//! data carries the fitted *non-constant* dispersion — not a single scalar.
//!
//! The bug: every dispersion-LS family fell into the scalar `else` branch of
//! `run_generate_unified`, dropping the whole `eta_d(x)` surface, so generated
//! data was homoscedastic at the seed dispersion (implied Gamma shape ~1 at
//! every x even though the true shape spanned ~6 → ~1). This mirrors the
//! correct `GaussianLocationScale` branch, which already threaded per-row sigma.
//!
//! This test fits a Gamma-LS model whose shape varies strongly with `x`
//! (`k(x) = exp(1.0 - 1.2 x)`, so the coefficient of variation `1/sqrt(k)` rises
//! steeply with `x`), generates many draws at a low and a high `x`, and asserts
//! the empirical CV ratio `CV(high)/CV(low)` tracks the true varying dispersion
//! (true ratio ~3.3). Pre-fix it was ~1.0 (constant). It is the user-facing
//! companion to the deterministic per-row variance-agreement test in
//! `bug_hunt_dispersion_location_scale_generate_predict_variance_agreement.rs`.

use std::path::Path;
use std::process::Command;

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits),
/// with a Marsaglia–Tsang Gamma draw, so the training set is byte-reproducible.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gamma(&mut self, k: f64, scale: f64) -> f64 {
        if k < 1.0 {
            let g = self.gamma(k + 1.0, scale);
            let u = self.unit().max(1e-300);
            return g * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * scale;
            }
        }
    }
}

fn write_training_csv(path: &Path) {
    let mut rng = Lcg(0x6125);
    let n = 3000usize;
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["y", "x"]).expect("write header");
    for i in 0..n {
        let x = -1.5 + 3.0 * (i as f64) / (n as f64 - 1.0);
        let mu = (0.6 + 0.4 * x).exp();
        let k = (1.0 - 1.2 * x).exp(); // Gamma shape; CV = 1/sqrt(k) rises with x
        let y = rng.gamma(k, mu / k);
        writer
            .write_record([format!("{y:.10}"), format!("{x:.10}")])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Read the `gam generate` output matrix: one row per input covariate row, one
/// column per draw (headers `draw_*`). Returns a row-major Vec<Vec<f64>>.
fn read_generate_matrix(path: &Path) -> Vec<Vec<f64>> {
    let mut reader = csv::Reader::from_path(path).expect("open generate csv");
    reader
        .records()
        .map(|rec| {
            rec.expect("generate row")
                .iter()
                .map(|v| v.parse::<f64>().expect("numeric draw"))
                .collect()
        })
        .collect()
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v)
}

#[test]
fn gamma_location_scale_generate_reproduces_per_row_precision() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("d.csv");
    let cov_path = dir.path().join("cov.csv");
    let model_path = dir.path().join("m.json");
    let out_path = dir.path().join("o.csv");

    write_training_csv(&train_path);

    // Fit Gamma location-scale: mean smooth on x, dispersion (log-precision)
    // smooth on x via --predict-noise. This is the #913/#1119 predictor.
    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg("y ~ smooth(x)")
        .args(["--predict-noise", "smooth(x)"])
        .args(["--family", "gamma-log"])
        .arg("--out")
        .arg(&model_path);
    let fit_out = fit_cmd.output().expect("spawn gam fit");
    assert!(
        fit_out.status.success(),
        "`gam fit` (gamma location-scale) failed.\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&fit_out.stdout),
        String::from_utf8_lossy(&fit_out.stderr),
    );

    // Generate at a low and a high x where the true shapes differ ~16x
    // (k(-1)=exp(2.2)=9.0 vs k(+1)=exp(-0.2)=0.82), so CV differs ~3.3x.
    let xs = [-1.0_f64, 1.0_f64];
    let mut w = csv::Writer::from_path(&cov_path).expect("create cov csv");
    w.write_record(["x"]).expect("cov header");
    for x in xs {
        w.write_record([format!("{x:.10}")]).expect("cov row");
    }
    w.flush().expect("flush cov csv");

    let mut gen_cmd = Command::new(gam::gam_binary!());
    gen_cmd
        .arg("generate")
        .arg(&model_path)
        .arg(&cov_path)
        .args(["--n-draws", "12000"])
        .args(["--seed", "20250615"])
        .arg("--out")
        .arg(&out_path);
    let gen_out = gen_cmd.output().expect("spawn gam generate");
    assert!(
        gen_out.status.success(),
        "`gam generate` failed.\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&gen_out.stdout),
        String::from_utf8_lossy(&gen_out.stderr),
    );

    let draws = read_generate_matrix(&out_path);
    assert_eq!(draws.len(), 2, "expected one output row per covariate row");

    let (m_lo, v_lo) = mean_var(&draws[0]);
    let (m_hi, v_hi) = mean_var(&draws[1]);
    let cv_lo = v_lo.sqrt() / m_lo;
    let cv_hi = v_hi.sqrt() / m_hi;
    let cv_ratio = cv_hi / cv_lo;
    // Implied Gamma shape k = mean^2 / var (CV = 1/sqrt(k)).
    let k_lo = m_lo * m_lo / v_lo;
    let k_hi = m_hi * m_hi / v_hi;

    eprintln!(
        "[gamma-LS generate #1125] x=-1: mean={m_lo:.3} CV={cv_lo:.3} impliedK={k_lo:.2} (true~9.0)\n\
         x=+1: mean={m_hi:.3} CV={cv_hi:.3} impliedK={k_hi:.2} (true~0.82)\n\
         CV(+1)/CV(-1)={cv_ratio:.3} (true ~3.3; pre-fix ~1.0)"
    );

    // The CV must rise steeply with x — pre-fix it was ~1.0 (constant dispersion,
    // per-row precision dropped). A 1.6 floor (vs true ~3.3) keeps headroom for
    // finite-sample / smoothing slack while decisively excluding the flat bug.
    assert!(
        cv_ratio > 1.6,
        "Gamma-LS generate did not reproduce the per-row precision: CV ratio {cv_ratio:.3} \
         (CV_lo={cv_lo:.3}, CV_hi={cv_hi:.3}); the eta_d(x) surface was dropped (#1125)"
    );
    // And the implied shape must drop with x in the right direction (k_hi < k_lo),
    // i.e. dispersion genuinely tracks the fitted surface rather than a constant.
    assert!(
        k_hi < 0.6 * k_lo,
        "implied Gamma shape did not fall with x (k_lo={k_lo:.2}, k_hi={k_hi:.2}); \
         dispersion looks constant (#1125)"
    );
}

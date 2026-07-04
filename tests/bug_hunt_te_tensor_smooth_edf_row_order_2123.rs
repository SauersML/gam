//! Bug hunt (#2123): a `te(x, z)` tensor-product smooth's reported EDF, standard
//! errors, and AIC depend on the ROW ORDER of the training frame. Rows of a
//! regression frame are exchangeable (the REML objective is a sum over rows), so
//! every fitted/inferential quantity must be invariant to a row permutation.
//!
//! Root cause: the tensor-product penalty log-determinant `log|S_λ|₊` is computed
//! from the Kronecker analytic marginal eigensystem, which is scaled
//! inconsistently with the marginal penalties actually assembled into the
//! Hessian `H = XᵀWX + S_λ`. At moderate λ the inconsistency is masked; at an
//! extreme-λ corner it makes the REML Occam term `½(log|H| − log|S_λ|₊)` swing
//! *negative* — which is mathematically impossible (Weyl: `H ⪰ S_λ` and X has
//! full column rank ⟹ the term is ≥ 0). The spurious low-cost basin at that
//! ill-conditioned corner traps the outer optimizer / seed grid on some row
//! orders (but not others), where the influence-EDF collapses and the fit is
//! falsely flagged non-converged, so the reported EDF is floored to the full
//! basis dimension and the SEs are rescaled.
//!
//! Observed before the fix (n=300, the exact numpy `default_rng(0)` frame from
//! the issue, dumped to `tests/data/te_2123_*.csv`):
//!   original order : edf ≈ 52.0  (railed floor)  Σ(SE) ≈ 3.84
//!   row permutation: edf ≈ 13.5  (converged)     Σ(SE) ≈ 12.98
//!
//! This test fits `te(x, z)` on the frame and on four fixed row permutations,
//! then asserts the deviances match (same data / same fit quality — an anchor)
//! and that the reported EDF agrees within a tight tolerance. It is RED before
//! the fix and GREEN once the outer REML converges robustly regardless of row
//! order.

use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn load_frame(name: &str) -> gam::data::EncodedDataset {
    let path = format!(
        "{}/tests/data/te_2123_{name}.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = text.lines();
    let header = lines.next().expect("header line");
    let headers: Vec<String> = header.split(',').map(String::from).collect();
    let mut rows: Vec<csv::StringRecord> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(csv::StringRecord::from(
            line.split(',').collect::<Vec<_>>(),
        ));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

struct FitSummary {
    edf_total: f64,
    deviance: f64,
    converged: bool,
    se_sum: f64,
}

fn fit_te(name: &str) -> FitSummary {
    let data = load_frame(name);
    let cfg = FitConfig::default();
    let FitResult::Standard(res) =
        fit_from_formula("y ~ te(x, z)", &data, &cfg).expect("te(x,z) fit should succeed")
    else {
        panic!("expected a standard GAM fit for te(x, z)");
    };
    let fit = &res.fit;
    let inference = fit.inference.as_ref().expect("inference present");
    let se_sum = inference
        .beta_covariance
        .as_ref()
        .map(|cov| {
            let cov = cov.as_array();
            (0..cov.nrows())
                .map(|j| cov[[j, j]].max(0.0).sqrt())
                .sum::<f64>()
        })
        .unwrap_or(f64::NAN);
    FitSummary {
        edf_total: inference.edf_total,
        deviance: fit.deviance,
        converged: fit.outer_converged,
        se_sum,
    }
}

#[test]
fn te_tensor_smooth_edf_is_row_order_invariant_2123() {
    let base = fit_te("orig");
    eprintln!(
        "original     : edf={:7.3} deviance={:.4} converged={} Σ(SE)={:.2}",
        base.edf_total, base.deviance, base.converged, base.se_sum
    );

    for seed in ["perm1", "perm7", "perm101", "perm2024"] {
        let f = fit_te(seed);
        eprintln!(
            "{seed:>9}: edf={:7.3} deviance={:.4} converged={} Σ(SE)={:.2}",
            f.edf_total, f.deviance, f.converged, f.se_sum
        );

        // Anchor: the permutation is an exchangeable relabeling, so the fit
        // quality (deviance) must match to numerical noise.
        assert!(
            (f.deviance - base.deviance).abs() < 1e-2,
            "deviance changed under a row permutation ({seed}): {} vs {}",
            f.deviance,
            base.deviance
        );

        // The reported EDF must be invariant to the row permutation.
        assert!(
            (f.edf_total - base.edf_total).abs() < 2.0,
            "reported EDF depends on row order ({seed}): original={:.3} vs perm={:.3} \
             (Σ(SE) {:.2} vs {:.2}) — the te tensor-product REML is not row-order invariant (#2123)",
            base.edf_total,
            f.edf_total,
            base.se_sum,
            f.se_sum
        );
    }
}

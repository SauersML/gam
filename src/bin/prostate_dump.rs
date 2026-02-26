#![allow(clippy::type_complexity)]

use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec,
    build_bspline_basis_1d,
};
use gam::estimate::{FitOptions, fit_gam, predict_gam};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};
use std::fs;

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn load_prostate_dataset() -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let raw = include_str!("../../benchmarks/datasets/prostate.csv");
    let mut pc1 = Vec::<f64>::new();
    let mut pc2 = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let pc1_i = parts
            .next()
            .ok_or_else(|| format!("prostate parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid pc1 at line {}: {e}", line_no + 1))?;
        let pc2_i = parts
            .next()
            .ok_or_else(|| format!("prostate parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid pc2 at line {}: {e}", line_no + 1))?;
        let y_i = parts
            .next()
            .ok_or_else(|| format!("prostate parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid y at line {}: {e}", line_no + 1))?;
        pc1.push(pc1_i);
        pc2.push(pc2_i);
        y.push(y_i);
    }
    if pc1.is_empty() {
        return Err("prostate dataset is empty".to_string());
    }
    Ok((
        Array1::from_vec(pc1),
        Array1::from_vec(pc2),
        Array1::from_vec(y),
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = parse_string_arg(&args, "--out")
        .unwrap_or_else(|| "benchmarks/prostate_rust_fitted.csv".to_string());

    let (pc1, pc2, y) = load_prostate_dataset().expect("failed to load prostate dataset");
    let n = pc1.len();

    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knot_spec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(8),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: true,
        identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
    };
    let built =
        build_bspline_basis_1d(pc2.view(), &spec).expect("failed to build prostate B-spline basis");
    let q = built.design.ncols();

    let p_total = q + 2;
    let mut x = Array2::<f64>::zeros((n, p_total));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = pc1[i];
        for j in 0..q {
            x[[i, j + 2]] = built.design[[i, j]];
        }
    }

    let s_list = built
        .penalties
        .iter()
        .map(|s_small| {
            let mut s_full = Array2::<f64>::zeros((p_total, p_total));
            for i in 0..q {
                for j in 0..q {
                    s_full[[i + 2, j + 2]] = s_small[[i, j]];
                }
            }
            s_full
        })
        .collect::<Vec<_>>();
    let nullspace_dims = built
        .nullspace_dims
        .iter()
        .map(|d| d + 2)
        .collect::<Vec<_>>();

    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let opts = FitOptions {
        max_iter: 200,
        tol: 1e-4,
        nullspace_dims,
    };
    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialLogit,
        &opts,
    )
    .expect("fit_gam failed");

    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialLogit,
    )
    .expect("predict_gam failed");

    let mut out = String::from("pc1,pc2,y,pred\n");
    for i in 0..n {
        out.push_str(&format!(
            "{},{},{},{}\n",
            pc1[i], pc2[i], y[i], pred.mean[i]
        ));
    }
    fs::write(&out_path, out).expect("failed to write output");
    eprintln!("Wrote {}", out_path);
}

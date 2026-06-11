//! SCRATCH (not for commit): diagnose gam's `re` linear random-slope on the
//! sleepstudy forecast — fit speed, RMSE, and any panic.
use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::path::Path;

const CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/sleepstudy.csv");

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

#[test]
fn re_diag() {
    init_parallelism();
    let text = std::fs::read_to_string(Path::new(CSV)).unwrap();
    let mut subjects: Vec<String> = Vec::new();
    let mut rows: Vec<(f64, String, f64)> = Vec::new();
    for (li, line) in text.lines().enumerate() {
        if li == 0 || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.trim().split(',').collect();
        let subj = c[3].to_string();
        if !subjects.contains(&subj) {
            subjects.push(subj.clone());
        }
        rows.push((c[2].parse().unwrap(), subj, c[1].parse().unwrap()));
    }
    let mut train = Vec::<StringRecord>::new();
    let mut tg = Vec::<usize>::new();
    let mut tx = Vec::<f64>::new();
    let mut tt = Vec::<f64>::new();
    for (days, subj, react) in &rows {
        let si = subjects.iter().position(|s| s == subj).unwrap();
        if *days <= 7.0 {
            train.push(StringRecord::from(vec![
                format!("{days:.17e}"),
                format!("S{subj}"),
                format!("{react:.17e}"),
            ]));
        } else {
            tg.push(si);
            tx.push(*days);
            tt.push(*react);
        }
    }
    let n_test = tt.len();
    assert_eq!(n_test, 36);
    let headers = vec!["Days".into(), "Subject".into(), "Reaction".into()];
    let ds = encode_recordswith_inferred_schema(headers, train).unwrap();
    let col = ds.column_map();
    let (dx, sx) = (col["Days"], col["Subject"]);
    let cfg = FitConfig {
        family: Some("gaussian".into()),
        ..FitConfig::default()
    };
    for f in [
        "Reaction ~ Days + s(Days, Subject, bs=\"re\")",
        "Reaction ~ s(Days, Subject, bs=\"re\")",
    ] {
        let t0 = std::time::Instant::now();
        match fit_from_formula(f, &ds, &cfg) {
            Ok(FitResult::Standard(fit)) => {
                let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
                let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
                for (r, (&g, &x)) in tg.iter().zip(&tx).enumerate() {
                    grid[[r, dx]] = x;
                    grid[[r, sx]] = g as f64;
                }
                match build_term_collection_design(grid.view(), &fit.resolvedspec) {
                    Ok(d) => {
                        let p: Vec<f64> = d.design.apply(&fit.fit.beta).to_vec();
                        eprintln!(
                            "[re-diag] {f:<48} rmse={:.3} edf={edf:.2} secs={:.1}",
                            rmse(&p, &tt),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                    Err(e) => eprintln!(
                        "[re-diag] {f:<48} PREDICT-ERR {e:?} secs={:.1}",
                        t0.elapsed().as_secs_f64()
                    ),
                }
            }
            Ok(_) => eprintln!("[re-diag] {f:<48} non-standard"),
            Err(e) => eprintln!(
                "[re-diag] {f:<48} FIT-ERR {e:?} secs={:.1}",
                t0.elapsed().as_secs_f64()
            ),
        }
    }
}

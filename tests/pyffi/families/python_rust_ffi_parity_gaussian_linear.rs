use csv::StringRecord;
use gam::inference::data::encode_recordswith_inferred_schema;
use gam::solver::fit_orchestration::{FitConfig, FitResult, fit_model, materialize};
use serde_json::Value;
use std::process::Command;

#[test]
fn python_rust_ffi_parity_gaussian_linear_case() {
    let n = 80usize;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        let y = 0.5 + 1.25 * x;
        rows.push(StringRecord::from(vec![x.to_string(), y.to_string()]));
    }
    let headers = vec!["x".to_string(), "y".to_string()];
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mat = materialize("y ~ x", &data, &cfg).expect("materialize");
    let fit = fit_model(mat.request).expect("fit");
    let FitResult::Standard(sf) = fit else {
        panic!("expected standard fit")
    };

    let beta_rust: Vec<f64> = sf.fit.beta.to_vec();
    let ll_rust = sf.fit.log_likelihood;
    let reml_rust = sf.fit.reml_score;

    let py = r#"
import json, math
import gamfit
rows=[]
n=80
for i in range(n):
    x=-1.0+2.0*i/(n-1)
    y=0.5+1.25*x
    rows.append({'x':x,'y':y})
m=gamfit.fit(rows, 'y ~ x', family='gaussian')
s=m.summary()
p=m.predict(rows, interval=0.95)
out={
 'beta': [c['estimate'] for c in s['coefficients']],
 'll': float(s['deviance']) if 'deviance' in s else float('nan'),
 'reml': float(s['reml_score']),
 'mean0': float(p[0]['.mean'] if isinstance(p, list) else p['.mean'][0]),
}
print(json.dumps(out))
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .output()
        .expect("run python");
    assert!(
        out.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    let beta_py: Vec<f64> = v["beta"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let reml_py = v["reml"].as_f64().unwrap();

    assert_eq!(beta_rust.len(), beta_py.len());
    for (i, (a, b)) in beta_rust.iter().zip(beta_py.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-9, "beta[{i}] rust={a} py={b}");
    }
    assert!(
        (reml_rust - reml_py).abs() <= 1e-9,
        "reml rust={reml_rust} py={reml_py}"
    );
    // The Python summary emits deviance (not the log-likelihood), so there is no
    // direct parity counterpart for `ll_rust`; assert the fitted Gaussian
    // log-likelihood is at least finite rather than silently discarding it.
    assert!(
        ll_rust.is_finite(),
        "rust gaussian log-likelihood must be finite: {ll_rust}"
    );
}

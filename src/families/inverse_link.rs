/// Apply a closed-form inverse link element-wise. Mirrors the family-kind
/// tags emitted by the Python-side family metadata.
pub fn apply_inverse_link_vec(eta: &[f64], family_kind: &str) -> Result<Vec<f64>, String> {
    let kind = family_kind.trim().to_ascii_lowercase();
    let mut out = Vec::with_capacity(eta.len());
    match kind.as_str() {
        "" | "identity" => out.extend_from_slice(eta),
        "logit" => {
            for &e in eta {
                out.push(if e >= 0.0 {
                    1.0 / (1.0 + (-e).exp())
                } else {
                    let ex = e.exp();
                    ex / (1.0 + ex)
                });
            }
        }
        "probit" => {
            let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
            for &e in eta {
                out.push(0.5 * (1.0 + statrs::function::erf::erf(e * inv_sqrt2)));
            }
        }
        "cloglog" => {
            for &e in eta {
                let clamped = e.clamp(-50.0, 50.0);
                out.push(1.0 - (-clamped.exp()).exp());
            }
        }
        "log" => {
            for &e in eta {
                out.push(e.exp());
            }
        }
        other => {
            return Err(format!(
                "posterior fitted-mean draws on response scale are not wired for \
                 family_kind={other:?}; access posterior.predict_draws(...).eta \
                 for link-scale draws or use model.predict(new_data, interval=...) \
                 for class-specific bands."
            ));
        }
    }
    Ok(out)
}

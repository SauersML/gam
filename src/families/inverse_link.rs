/// Apply a closed-form inverse link element-wise from a string tag.
///
/// Kept as string-dispatched (rather than routed through the canonical
/// `InverseLink` enum) because the two callers — `inference::eta_bands`
/// and `inference::posterior_bands` — are reached only through the
/// Python FFI (`crates/gam-pyffi`), which hands in `family_kind` as a
/// `&str` carrying the family metadata produced on the Python side.
/// No `InverseLink` value is in scope at those entry points; constructing
/// one would require re-parsing the same string into the parameterized
/// variants (`Sas`, `Mixture`, `LatentCLogLog`, ...) whose state these
/// posterior-band callers do not have access to either. The supported
/// kinds here are the closed-form `Standard` links plus `identity`, which
/// is the full set the FFI surface promises for posterior-band quantiles.
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
            // μ(η) = 1 − exp(−exp(η)). The naive `1.0 − exp(−exp(η))`
            // form loses every digit in the deep negative tail: as soon
            // as exp(−exp(η)) rounds to 1.0 (η ≲ −36 in f64), the
            // subtraction collapses to exactly 0.0, breaking strict
            // positivity and monotonicity of μ in η. `expm1` evaluates
            // exp(x) − 1 without the cancellation, so
            //   μ = 1 − exp(−exp(η)) = −expm1(−exp(η))
            // is exact all the way down to the exp() underflow boundary
            // (η ≈ −745). Matches `cloglog_negative_tail_mean` in
            // `inference/quadrature.rs` and the `CLogLog` jet in
            // `solver/mixture_link.rs`. See issue #344.
            for &e in eta {
                let clamped = e.clamp(-50.0, 50.0);
                out.push(-((-clamped.exp()).exp_m1()));
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

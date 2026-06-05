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
                // Φ(η) = ½·erfc(−η/√2). The naive ½(1+erf(η/√2)) form cancels
                // in the deep negative tail (erf saturates at −1.0 for η ≲ −8.3,
                // collapsing Φ to exactly 0.0); erfc is a dedicated complementary
                // tail evaluator and stays accurate to the f64 underflow boundary.
                out.push(0.5 * statrs::function::erf::erfc(-e * inv_sqrt2));
            }
        }
        "cloglog" => {
            for &e in eta {
                // μ = 1 − exp(−exp(η)); use -expm1(-exp(η)) to preserve precision
                // in the deep negative tail where exp(-exp(η)) rounds to 1.0 and
                // the naive `1.0 - …` form collapses to 0 for η ≲ -36.
                //
                // No clamp on η: the -expm1(-exp(η)) form is finite, accurate, and
                // strictly monotone across the entire representable range, so the
                // old [-50, 50] clamp was both unnecessary and actively wrong.
                //   • Positive tail: exp(η) overflows to +∞ only for η ≳ 709.8,
                //     where expm1(-∞) = -1 yields μ = 1 exactly — the correct limit
                //     (μ already rounds to 1.0 well before η = 50, so the old +50
                //     bound was merely redundant).
                //   • Negative tail: exp(η) underflows to 0 only for η ≲ -745.1,
                //     where expm1(0) = 0 yields μ = 0 exactly — the correct limit.
                //     Above that, μ tracks exp(η) faithfully. The old -50 bound
                //     froze μ at exp(-50) ≈ 1.93e-22 for every η ≤ -50, destroying
                //     strict monotonicity and the leading-order μ(η) ~ exp(η)
                //     asymptotic that the sibling logit/probit branches preserve.
                out.push(-(-e.exp()).exp_m1());
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

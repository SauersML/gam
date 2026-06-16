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
                // Canonical EXACT public log inverse link: bare `exp(η)` with the
                // correct IEEE semantics — finite wherever `exp(η)` is representable,
                // `0.0` on underflow (η ≲ −745.1), `+∞` on overflow (η ≳ 709.8).
                //
                // Deliberately NO `η.clamp(−700, 700)` here. That clamp lives only
                // in the SOLVER's inverse-link jet (`LinkFunction::Log` in
                // `solver/mixture_link.rs`) and PIRLS working-state engine
                // (`solver/pirls/log_link_working_state.rs::ETA_CLAMP`), where it is
                // an intentional conditioning hack that keeps the IRLS normal
                // equations well posed. On FINITE η the two disagree: e.g. at
                // η = 705 the exact value is exp(705) ≈ 1.5e306 (finite and correct)
                // while the clamped solver value is exp(700) ≈ 1.0e304 (off by
                // exp(5) ≈ 148×); at η = −720 the exact value underflows toward 0
                // (≈ 2e−313) while the clamp pins it at exp(−700) ≈ 9.9e−305. Public
                // response-scale outputs (FFI `apply_inverse_link_array`, posterior
                // bands) must report the exact value, so they route here, never
                // through the solver's clamped jet.
                out.push(e.exp());
            }
        }
        other => {
            return Err(format!(
                "posterior fitted-mean draws on response scale are not wired for \
                 family_kind={other:?} from the bare string tag; the parameterized \
                 links (sas, mixture, latent-cloglog, beta-logistic) carry per-fit \
                 state and must be routed through the serialized `link_spec` (see \
                 `apply_inverse_link_spec_vec`). access posterior.predict_draws(...).eta \
                 for link-scale draws or use model.predict(new_data, interval=...) \
                 for class-specific bands."
            ));
        }
    }
    Ok(out)
}

/// Apply a closed-form inverse link element-wise from a fully parameterized
/// [`InverseLink`] spec.
///
/// This is the typed companion to [`apply_inverse_link_vec`]. The string-tag
/// entry point cannot evaluate the *parameterized* links — `Sas`, `Mixture`,
/// `LatentCLogLog`, `BetaLogistic` — because their fitted state (skew/tail,
/// mixture weights, latent SD) is lost the moment the family is collapsed to a
/// bare `&str`. The Python FFI now carries the serialized `InverseLink` through
/// the sample payload as `link_spec`, so this function reconstructs the exact
/// response-scale mean `μ = g⁻¹(η)` for every supported link by routing through
/// the canonical solver evaluator `inverse_link_mu_d1_for_inverse_link`.
///
/// The returned `μ` is bit-identical to the `mu` field of the solver's full
/// inverse-link jet for the parameterized links, and matches
/// [`apply_inverse_link_vec`] exactly for the `Standard` links *except* for the
/// `Log` link: the public response transform reports the EXACT `exp(η)` (see the
/// `LinkFunction::Log` branch above and issue #963), whereas the solver jet
/// applies the conditioning clamp `η.clamp(−700, 700)`. To preserve that public
/// contract this routes `Standard(Log)` (and `Standard(Identity)`) through the
/// string path, and only the genuinely parameterized / non-log links through the
/// solver evaluator.
pub fn apply_inverse_link_spec_vec(
    eta: &[f64],
    link: &crate::types::InverseLink,
) -> Result<Vec<f64>, String> {
    use crate::types::{InverseLink, StandardLink};

    // Standard links have a documented EXACT public response transform (notably
    // the unclamped `exp(η)` for Log) that diverges from the solver's clamped
    // jet on finite boundary η. Keep them on the string path so the public
    // contract pinned by `public_log_inverse_link_is_exact_exp_not_solver_clamp`
    // is preserved regardless of which entry point is used.
    if let InverseLink::Standard(std_link) = link {
        let tag = match std_link {
            StandardLink::Identity => "identity",
            StandardLink::Log => "log",
            StandardLink::Logit => "logit",
            StandardLink::Probit => "probit",
            StandardLink::CLogLog => "cloglog",
        };
        return apply_inverse_link_vec(eta, tag);
    }

    let mut out = Vec::with_capacity(eta.len());
    for &e in eta {
        let (mu, _d1) =
            crate::solver::mixture_link::inverse_link_mu_d1_for_inverse_link(link, e)
                .map_err(|err| {
                    format!("failed to evaluate parameterized inverse link at eta={e}: {err}")
                })?;
        out.push(mu);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::apply_inverse_link_vec;

    /// The public log inverse link is the EXACT `exp(η)`, never the solver's
    /// `η.clamp(−700, 700).exp()` conditioning transform. This pins the contract
    /// at the finite boundary η values where the two implementations diverge, so
    /// a future refactor cannot silently reroute a public response-scale output
    /// through the clamped solver jet (issue #963).
    #[test]
    fn public_log_inverse_link_is_exact_exp_not_solver_clamp() {
        // η = 705: exact exp(705) is finite (≈1.5e306); the solver clamp would
        // return exp(700) (≈1.0e304), wrong by a factor of exp(5) ≈ 148.
        let out = apply_inverse_link_vec(&[705.0], "log").expect("log inverse link");
        assert_eq!(out.len(), 1);
        let exact = 705.0_f64.exp();
        assert!(exact.is_finite(), "exp(705) must be representable in f64");
        assert_eq!(
            out[0], exact,
            "public log inverse link must be exact exp(705), not the solver clamp"
        );
        let clamped = 700.0_f64.exp();
        assert!(
            out[0] > clamped * 100.0,
            "exact exp(705) must exceed the clamped exp(700) by ~exp(5); got {} vs {}",
            out[0],
            clamped
        );

        // η = −720: exact exp(−720) underflows toward 0 (≈2e−313, subnormal);
        // the solver clamp would pin it at exp(−700) ≈ 9.9e−305, ~4.85e8× too
        // large. Either way the exact value is strictly below the clamped one.
        let out = apply_inverse_link_vec(&[-720.0], "log").expect("log inverse link");
        let exact = (-720.0_f64).exp();
        assert_eq!(
            out[0], exact,
            "public log inverse link must be exact exp(-720), not the solver clamp"
        );
        let clamped = (-700.0_f64).exp();
        assert!(
            out[0] < clamped,
            "exact exp(-720) must be strictly below the clamped exp(-700); got {} vs {}",
            out[0],
            clamped
        );

        // True IEEE overflow/underflow limits are honored exactly.
        let over = apply_inverse_link_vec(&[710.0], "log").expect("log inverse link");
        assert!(
            over[0].is_infinite() && over[0] > 0.0,
            "exp(710) overflows to +inf under the exact public transform"
        );
        let under = apply_inverse_link_vec(&[-746.0], "log").expect("log inverse link");
        assert_eq!(
            under[0], 0.0,
            "exp(-746) underflows to exactly 0.0 under the exact public transform"
        );
    }
}

//! Necessary scope bound when an omitted intervention forces the same explained distribution.
//! The JS theorem is exact for the fixed native teacher distributions. Reported numerical
//! intervals are conditional on acceptance::kl_logits' comparison rounding model, including
//! its transcendental assumptions; they are not a proved Rust/libm or full-network certificate.
use super::acceptance::kl_logits;
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScopeBound {
    pub clean_tokens: usize,
    pub common_suffix_tokens: usize,
    pub clean_group_episodes: usize,
    pub edited_group_episodes: usize,
    pub coefficient: f64,
    pub mean_js: f64,
    pub conditional_comparison_error: f64,
    pub conditional_js_lower: f64,
    pub conditional_js_upper: f64,
    pub conditional_worst_group_lower: f64,
    pub condition: &'static str,
}
fn normalized(z: ArrayView1<'_, f64>) -> Result<(Array1<f64>, f64), String> {
    if z.is_empty() || z.iter().any(|v| !v.is_finite()) {
        return Err("nonempty finite teacher logits required".into());
    }
    let maximum = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let terms: Vec<f64> = z.iter().map(|v| (v - maximum).exp()).collect();
    if terms.iter().any(|v| *v == 0.0 || !v.is_finite()) {
        return Err(
            "diagnostic comparison model does not cover underflowed teacher exponentials".into(),
        );
    }
    let sum: f64 = terms.iter().sum();
    let log_sum = sum.ln();
    let normalizer = maximum + log_sum;
    let error =
        (accumulation_growth(z.len() + 4) * (maximum.abs() + log_sum.abs() + 1.0)).next_up();
    Ok((z.mapv(|v| v - normalizer), error))
}
fn js_row(p: ArrayView1<'_, f64>, r: ArrayView1<'_, f64>) -> Result<(f64, f64), String> {
    if p.len() != r.len() {
        return Err("teacher class widths differ".into());
    }
    let (lp, ep) = normalized(p)?;
    let (lr, er) = normalized(r)?;
    let mut mixture = Array1::zeros(p.len());
    let mut delta = 0.0_f64;
    for ((out, &a), &b) in mixture.iter_mut().zip(lp.iter()).zip(lr.iter()) {
        let m = a.max(b);
        let (ea, eb) = ((a - m).exp(), (b - m).exp());
        if ea == 0.0 || eb == 0.0 {
            return Err(
                "diagnostic comparison model does not cover underflowed mixture exponentials"
                    .into(),
            );
        }
        let l = (ea + eb).ln();
        let log_sum = m + l;
        *out = log_sum - std::f64::consts::LN_2;
        // logaddexp is 1-Lipschitz in its inputs in the infinity norm. The last term
        // accounts for the log(2) constant and subtraction under the same nominal model.
        let construction = (accumulation_growth(6) * (m.abs() + l.abs() + 1.0)).next_up();
        let subtraction =
            (accumulation_growth(2) * (log_sum.abs() + std::f64::consts::LN_2 + 1.0)).next_up();
        delta = delta.max((ep.max(er) + construction + subtraction).next_up());
    }
    let (a, ea) = kl_logits(p, mixture.view());
    let (b, eb) = kl_logits(r, mixture.view());
    let value = (a + b) * 0.5;
    // A sup-norm logit perturbation delta changes each normalized log-probability
    // by at most 2*delta. Include that mixture error in addition to the KL comparison.
    let error =
        ((ea + eb) * 0.5 + 2.0 * delta + accumulation_growth(2) * (a.abs() + b.abs())).next_up();
    if !value.is_finite() || !error.is_finite() {
        return Err("nonfinite JS diagnostic".into());
    }
    Ok((value, error))
}
/// `clean_suffix` and `edited_suffix` must contain exactly the same native input token rows.
/// The caller declares clean-group averaging over T tokens and m episodes, edited-group
/// averaging over K suffix tokens and n episodes. Other losses are nonnegative.
/// The exact necessary bound is 2K/(mT+nK) times mean_suffix_JS, provided the explanation
/// predicts the identical Q on these clean and edited inputs. No candidate is pruned here.
pub fn omitted_intervention_bound(
    clean_suffix: &Array2<f64>,
    edited_suffix: &Array2<f64>,
    clean_tokens: usize,
    clean_group_episodes: usize,
    edited_group_episodes: usize,
) -> Result<ScopeBound, String> {
    let k = clean_suffix.nrows();
    if clean_suffix.dim() != edited_suffix.dim()
        || k == 0
        || clean_tokens < k
        || clean_group_episodes == 0
        || edited_group_episodes == 0
    {
        return Err("matching nonempty suffix domains and positive group sizes required".into());
    }
    let numerator = k.checked_mul(2).ok_or("coefficient overflow")?;
    let denominator = clean_group_episodes
        .checked_mul(clean_tokens)
        .and_then(|x| {
            edited_group_episodes
                .checked_mul(k)
                .and_then(|y| x.checked_add(y))
        })
        .ok_or("coefficient overflow")?;
    if numerator as u128 > (1_u128 << 53) || denominator as u128 > (1_u128 << 53) {
        return Err("coefficient integer domain exceeds exact binary64 representation".into());
    }
    let coefficient = numerator as f64 / denominator as f64;
    let (mut sum, mut errors, mut magnitude) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (p, r) in clean_suffix.outer_iter().zip(edited_suffix.outer_iter()) {
        let (v, e) = js_row(p, r)?;
        sum += v;
        errors += e;
        magnitude += v.abs();
    }
    let mean_js = sum / k as f64;
    let conditional_comparison_error =
        ((errors + accumulation_growth(k + 4) * (magnitude + errors + sum.abs())) / k as f64)
            .next_up();
    let lower = (mean_js - conditional_comparison_error)
        .next_down()
        .max(0.0);
    let upper = (mean_js + conditional_comparison_error)
        .next_up()
        .min(std::f64::consts::LN_2.next_up());
    let bound = (coefficient.next_down() * lower).next_down().max(0.0);
    Ok(ScopeBound {
        clean_tokens,
        common_suffix_tokens: k,
        clean_group_episodes,
        edited_group_episodes,
        coefficient,
        mean_js,
        conditional_comparison_error,
        conditional_js_lower: lower,
        conditional_js_upper: upper,
        conditional_worst_group_lower: bound,
        condition: "same explained Q on matched clean/edited input rows; other episode losses nonnegative; interval conditional on existing KL comparison rounding assumptions, not libm or whole-network proof",
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn binary(a: f64) -> Array2<f64> {
        Array2::from_shape_vec((1, 2), vec![a.ln(), (1.0 - a).ln()]).unwrap()
    }
    #[test]
    fn exact_binary_js_is_inside_conditional_interval() {
        let b = omitted_intervention_bound(&binary(0.8), &binary(0.2), 1, 2, 1).unwrap();
        let exact = std::f64::consts::LN_2 + 0.8_f64 * 0.8_f64.ln() + 0.2_f64 * 0.2_f64.ln();
        assert!(b.conditional_js_lower <= exact && exact <= b.conditional_js_upper);
        assert_eq!(b.coefficient, 2.0 / 3.0);
        assert!(b.conditional_worst_group_lower > 0.12);
    }
    #[test]
    fn actual_suffix_weight_is_not_full_clean_weight() {
        let p = binary(0.8);
        let r = binary(0.2);
        let suffix = omitted_intervention_bound(&p, &r, 16, 2, 1).unwrap();
        let matched = omitted_intervention_bound(&p, &r, 1, 2, 1).unwrap();
        assert_eq!(suffix.coefficient, 2.0 / 33.0);
        assert!(
            suffix.conditional_worst_group_lower < matched.conditional_worst_group_lower / 10.0
        );
        let wrong = Array2::zeros((2, 2));
        assert!(omitted_intervention_bound(&p, &wrong, 16, 2, 1).is_err());
    }
    #[test]
    fn identical_teachers_do_not_fabricate_positive_floor() {
        let p = binary(0.8);
        let b = omitted_intervention_bound(&p, &p, 16, 2, 1).unwrap();
        assert_eq!(b.conditional_js_lower, 0.0);
        assert_eq!(b.conditional_worst_group_lower, 0.0);
    }
    #[test]
    fn unsupported_numerics_and_domains_fail_conservatively() {
        assert!(omitted_intervention_bound(&binary(0.5), &binary(0.5), 0, 2, 1).is_err());
        let p = Array2::from_shape_vec((1, 2), vec![0.0, -1000.0]).unwrap();
        assert!(omitted_intervention_bound(&p, &p, 1, 2, 1).is_err());
    }
}

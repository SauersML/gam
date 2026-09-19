//! Whether the restart from a certified strict saddle's escape point searches on
//! the declared analytic Hessian (#2939) or keeps the gradient-only search the
//! route prefers (#2898), decided at the escape point (#2954 stage 1b).

use super::bridges::projected_gradient_norm;
use super::run::{OuterConfig, outer_gradient_tolerance, outer_search_bounds_template};
use super::{OuterEvalOrder, OuterObjective};
use ndarray::Array1;

/// #2939 latched the curvature search after every certified saddle because its
/// escape point still sat inside the solver's gradient band, where a
/// gradient-only restart stops at iteration 0 and the next mint finds the
/// saddle again. That premise is a measurement at the escape point, so this
/// takes it there: one value-and-gradient evaluation, projected on the search
/// box, against the band the restart's search stops on.
///
/// Inside the band, or where the point cannot be evaluated, the restart
/// latches, as #2939 does. Outside it the restart keeps the gradient-only
/// search, and the mint still prices the exact Hessian once. On gnomon#2359's
/// 200-row BMS fit the ρ = −2 saddle's escape point sits at `|g| = 2.3e-2`
/// against a band of `4.0e-3`: a latched ARC restart paid the exact outer
/// Hessian at every step (62-79 s each) and did not certify within 900 s, and a
/// BFGS restart certified 126.6611 in the lower basin. An earlier escape in the
/// same solve means a gradient-only restart already failed to leave a saddle, so
/// every later one latches.
pub(super) fn saddle_escape_needs_curvature_search(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    escape: Option<&Array1<f64>>,
    earlier_escapes: usize,
) -> bool {
    let Some(escape) = escape else {
        return true;
    };
    if earlier_escapes > 0 {
        log::info!(
            "[OUTER] {context}: saddle escape {} of this solve; a gradient-only restart \
             already failed to leave a saddle, so the search latches the declared Hessian \
             (#2939)",
            earlier_escapes + 1,
        );
        return true;
    }
    let band = outer_gradient_tolerance(config).abs;
    let measured = obj
        .eval_with_order(escape, OuterEvalOrder::ValueAndGradient)
        .ok()
        .filter(|eval| {
            eval.cost.is_finite()
                && eval.gradient.len() == escape.len()
                && eval.gradient.iter().all(|value| value.is_finite())
        })
        .map(|eval| {
            let bounds = outer_search_bounds_template(config, escape.len());
            projected_gradient_norm(escape, &eval.gradient, Some(&bounds))
        });
    match measured {
        Some(norm) if norm > band => {
            log::info!(
                "[OUTER] {context}: the saddle escape point's |Pg|={norm:.3e} clears the \
                 solver band {band:.3e}, so the restart keeps the gradient-only search and \
                 the mint prices the declared Hessian (#2898, #2954)"
            );
            false
        }
        Some(norm) => {
            log::info!(
                "[OUTER] {context}: the saddle escape point's |Pg|={norm:.3e} is inside the \
                 solver band {band:.3e}; a gradient-only restart would stop at iteration 0, \
                 so the search latches the declared Hessian (#2939)"
            );
            true
        }
        None => {
            log::info!(
                "[OUTER] {context}: the saddle escape point did not evaluate to a finite \
                 gradient; the search latches the declared Hessian (#2939)"
            );
            true
        }
    }
}

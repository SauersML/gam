//! Shared independent AD oracles, available only to function-prior tests.
use super::*;

pub(super) fn category_oracle<S: JetField>(prior: &CategoryPriors, theta: &[S]) -> S {
    let mut total = theta[0].constant_like(0.0);
    for (intercept, gaps) in &prior.channels {
        let cuts = gaps.len() + 1;
        total = add_real(
            &total,
            (1..=cuts).map(|j| (j as f64).ln()).sum::<f64>()
                - 0.5 * cuts as f64 * (2.0 * std::f64::consts::PI).ln(),
        );
        let mut cut = theta[0].constant_like(0.0);
        for j in 0..cuts {
            if j > 0 {
                let q = &theta[gaps.start + j - 1];
                cut = cut.add(&emission::softplus(q));
                total = total.sub(&emission::softplus(&q.neg()));
            }
            let z = cut.sub(&theta[*intercept]);
            total = total.sub(&z.mul(&z).scale(0.5));
        }
    }
    total
}

pub(super) fn structural_oracle<S: JetField>(
    function: &StructuralFunction,
    theta: &[S],
    rho: &S,
) -> S {
    let (log_value, log_jacobian, shape, log_gamma) = match *function {
        StructuralFunction::CountMean { coordinate } => (
            theta[coordinate].clone(),
            theta[coordinate].clone(),
            1.0,
            0.0,
        ),
        StructuralFunction::TemporalVariation {
            coordinate,
            log_time,
        } => {
            let q = &theta[coordinate];
            (
                add_real(&ln(&emission::softplus(q)), log_time),
                add_real(&emission::softplus(&q.neg()).neg(), log_time),
                1.0,
                0.0,
            )
        }
        StructuralFunction::MeasurementPrecision { coordinate } => {
            let v = theta[coordinate].scale(-2.0);
            (v.clone(), add_real(&v, 2.0_f64.ln()), 3.0, 2.0_f64.ln())
        }
        StructuralFunction::InverseSoftplus {
            coordinate,
            log_multiplier,
        } => {
            let q = &theta[coordinate];
            let s = ln(&emission::softplus(q));
            (
                add_real(&s.neg(), log_multiplier),
                add_real(
                    &emission::softplus(&q.neg()).neg().sub(&s.scale(2.0)),
                    log_multiplier,
                ),
                5.0,
                24.0_f64.ln(),
            )
        }
    };
    add_real(
        &rho.scale(shape)
            .add(&log_value.scale(shape - 1.0))
            .add(&log_jacobian)
            .sub(&exp(&rho.add(&log_value))),
        -log_gamma,
    )
}

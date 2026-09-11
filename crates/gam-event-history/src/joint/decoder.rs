//! Normalized decoder weights shared by all states at one coefficient value.
use super::*;

pub(super) struct PreparedDecoder<S> {
    log_weights: Vec<S>,
    signatures: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::jet_scalar::Order1;
    use std::{hint::black_box, time::Instant};

    fn model(signatures: usize) -> JointLikelihood {
        JointLikelihood::new(JointSpecification {
            signatures,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap()
    }

    fn direct(model: &JointLikelihood, theta: &[f64], state: &[f64]) -> f64 {
        let mut numerator = vec![0.0];
        let mut denominator = vec![0.0];
        for (axis, x) in state.iter().enumerate() {
            let weight = theta[model.layout.decoder.start + axis];
            numerator.push(weight + emission::log_softplus(x));
            denominator.push(weight);
        }
        log_sum_exp(&numerator) - log_sum_exp(&denominator)
    }

    #[test]
    fn decoder_preserves_extreme_simplex_mass_and_matches_direct_log_sums() {
        for k in [0, 1, 2, 5] {
            let model = model(k);
            for scale in [0.0, 0.1, 1.0, 20.0, 800.0] {
                let mut theta = vec![0.0; model.layout.width];
                for axis in 0..k {
                    theta[model.layout.decoder.start + axis] = scale * ((axis + 1) as f64).sin();
                }
                let decoder = PreparedDecoder::new(&model, &theta);
                for n in 0..100 {
                    let state: Vec<_> = (0..k)
                        .map(|axis| scale * ((n + axis) as f64).cos())
                        .collect();
                    assert!(
                        (decoder.activity(0, &state) - direct(&model, &theta, &state)).abs()
                            < 5e-12
                    );
                }
            }
        }
        let model = model(2);
        let mut theta = vec![Order1::<1> { v: 0.0, g: [0.0] }; model.layout.width];
        for axis in 0..2 {
            theta[model.layout.decoder.start + axis].v = 1e300;
        }
        theta[model.layout.decoder.start].g[0] = 1.0;
        let decoder = PreparedDecoder::new(&model, &theta);
        let state = [
            Order1::<1> { v: -3.0, g: [0.0] },
            Order1::<1> { v: 2.0, g: [0.0] },
        ];
        let result = decoder.activity(0, &state);
        let a = emission::softplus(&(-3.0));
        let b = emission::softplus(&2.0);
        assert!((result.v - ((a + b) * 0.5).ln()).abs() < 1e-14);
        assert!((result.g[0] - (a / (a + b) - 0.5)).abs() < 1e-14);
    }

    #[test]
    fn prepared_decoder_speed() {
        let model = model(2);
        let mut theta = vec![0.0; model.layout.width];
        theta[model.layout.decoder.start] = -0.4;
        theta[model.layout.decoder.start + 1] = 0.7;
        let states: Vec<_> = (0..2048)
            .map(|i| vec![((i as f64) * 0.1).sin(), ((i as f64) * 0.2).cos()])
            .collect();
        let decoder = PreparedDecoder::new(&model, &theta);
        let start = Instant::now();
        let mut previous = 0.0;
        for _ in 0..32 {
            for state in &states {
                previous += black_box(direct(&model, &theta, black_box(state)));
            }
        }
        let old_time = start.elapsed();
        let start = Instant::now();
        let mut prepared = 0.0;
        for _ in 0..32 {
            for state in &states {
                prepared += black_box(decoder.activity(0, black_box(state)));
            }
        }
        let new_time = start.elapsed();
        assert!((previous - prepared).abs() < 1e-8);
        eprintln!(
            "decoder: direct={old_time:?}, prepared={new_time:?}, ratio={:.3}",
            old_time.as_secs_f64() / new_time.as_secs_f64()
        );
    }
}

impl<S: JetField> PreparedDecoder<S> {
    pub(super) fn new(model: &JointLikelihood, theta: &[S]) -> Self {
        let signatures = model.spec.signatures;
        let mut log_weights = Vec::with_capacity(model.spec.marks.len() * (signatures + 1));
        for d in 0..model.spec.marks.len() {
            let start = model.layout.decoder.start + d * signatures;
            let logits = &theta[start..start + signatures];
            let shift = logits.iter().map(|w| w.value()).fold(0.0_f64, f64::max);
            let mut sum = theta[0].constant_like((-shift).exp());
            for weight in logits {
                sum = sum.add(&exp(&add_real(weight, -shift)));
            }
            let log_sum = ln(&sum);
            log_weights.push(add_real(&log_sum.neg(), -shift));
            for weight in logits {
                // Subtract the large common shift before log(sum): equal
                // logits near 1e300 still share their mass, rather than each
                // acquiring a rounded log probability of zero.
                log_weights.push(add_real(weight, -shift).sub(&log_sum));
            }
        }
        Self {
            log_weights,
            signatures,
        }
    }

    pub(super) fn weights(&self, mark: usize) -> &[S] {
        let start = mark * (self.signatures + 1);
        &self.log_weights[start..start + self.signatures + 1]
    }

    pub(super) fn activity(&self, mark: usize, state: &[S]) -> S {
        let weights = self.weights(mark);
        let mut total = weights[0].clone();
        for (weight, x) in weights[1..].iter().zip(state) {
            let term = weight.add(&emission::log_softplus(x));
            total = if total.value() >= term.value() {
                total.add(&emission::softplus(&term.sub(&total)))
            } else {
                term.add(&emission::softplus(&total.sub(&term)))
            };
        }
        total
    }
}

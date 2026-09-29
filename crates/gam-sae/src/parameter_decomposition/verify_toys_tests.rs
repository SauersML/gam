#![cfg(test)]
//! Toy 6 of #2951 through exhaustive verification: two component reads that differ along a
//! direction the data never visits are indistinguishable on the data family for every
//! amount of that direction, and distinguishable off it, while their sum is the same map
//! everywhere.

use crate::parameter_decomposition::secant::BandedMatrix;
use crate::parameter_decomposition::supports::{EvidenceStatus, ExactBasis};
use crate::parameter_decomposition::test_support::planted_toys::{Ball, data_subspace_toy, dyadic};
use crate::parameter_decomposition::test_support::test_governor;
use crate::parameter_decomposition::verify::{
    CounterfactualContract, FamilyStatus, Tolerance, verify_counterfactual_contract,
};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// `D(t) x = u (r + t v)ᵀ x` as one logit row, with ball radii: the read, its inner product
/// with `x` and the write each carried by ball arithmetic.
fn component_logits(write: &Array1<f64>, read: &Array1<f64>, direction: &Array1<f64>, amount: f64, x: &Array1<f64>) -> BandedMatrix {
    let moved: Vec<Ball> = read
        .iter()
        .zip(direction.iter())
        .map(|(&r, &v)| Ball::exact(r).add(Ball::exact(amount).mul(Ball::exact(v))))
        .collect();
    let inner = Ball::sum(moved.iter().zip(x.iter()).map(|(&r, &value)| r.mul(Ball::exact(value))));
    let row: Vec<Ball> = write.iter().map(|&u| Ball::exact(u).mul(inner)).collect();
    BandedMatrix {
        values: Array2::from_shape_fn((1, row.len()), |(_, i)| row[i].value),
        bands: Array2::from_shape_fn((1, row.len()), |(_, i)| row[i].radius),
    }
}

struct Toy {
    write: Array1<f64>,
    first: Array1<f64>,
    second: Array1<f64>,
    direction: Array1<f64>,
    on_data: Vec<Array1<f64>>,
    off_data: Vec<Array1<f64>>,
}

fn toy() -> Toy {
    let planted = data_subspace_toy(61, 32);
    let mut rng = StdRng::seed_from_u64(62);
    let on_data = planted.data.rows().into_iter().map(|row| row.to_owned()).collect();
    let off_data = dyadic(&mut rng, 32, 8, 16, 8.0).rows().into_iter().map(|row| row.to_owned()).collect();
    let vectors = dyadic(&mut rng, 3, 8, 8, 8.0);
    Toy {
        write: vectors.row(0).to_owned(),
        first: vectors.row(1).to_owned(),
        second: vectors.row(2).to_owned(),
        direction: planted.hidden,
        on_data,
        off_data,
    }
}

const AMOUNTS: [f64; 3] = [0.0, 0.5, 2.0];

fn declared() -> Tolerance {
    Tolerance {
        kl: 1e-9,
        centred_logit_gap: 1e-9,
    }
}

fn assert_exact_zero(status: &FamilyStatus) {
    match status {
        EvidenceStatus::Exact { value, numerical_error, basis, .. } => {
            assert!(matches!(basis, ExactBasis::Exhaustive { .. }));
            assert!(value <= numerical_error, "{value} beyond its band {numerical_error}");
        }
        other => panic!("expected an exhaustive exact status, got {other:?}"),
    }
}

/// `D₁(t) = u (a + t v)ᵀ` against `D₁(0)` over the amounts `t`, on one input family.
fn first_component(toy: &Toy, family: &[Array1<f64>]) -> CounterfactualContract {
    let reference = |_: &MemoryGovernor, _: &f64, x: &Array1<f64>| -> Result<BandedMatrix, std::fmt::Error> {
        Ok(component_logits(&toy.write, &toy.first, &toy.direction, 0.0, x))
    };
    let candidate = |_: &MemoryGovernor, amount: &f64, x: &Array1<f64>| -> Result<BandedMatrix, std::fmt::Error> {
        Ok(component_logits(&toy.write, &toy.first, &toy.direction, *amount, x))
    };
    verify_counterfactual_contract(test_governor(), &reference, &candidate, &AMOUNTS, family, &declared()).expect("verification")
}

#[test]
fn toy6_components_differ_only_off_the_data() {
    let toy = toy();
    let on = first_component(&toy, &toy.on_data);
    assert!(on.certified_within(&declared()));
    for family in &on.interventions {
        for status in [&family.forward_kl, &family.reverse_kl, &family.centred_logit_gap] {
            assert_exact_zero(status);
        }
    }
    let off = first_component(&toy, &toy.off_data);
    assert!(!off.certified_within(&declared()));
    for status in [&off.interventions[0].forward_kl, &off.interventions[0].centred_logit_gap] {
        assert_exact_zero(status);
    }
    for family in &off.interventions[1..] {
        assert!(matches!(family.centred_logit_gap, EvidenceStatus::Counterexample { .. }), "{:?}", family.centred_logit_gap);
    }

    // The sum `D₁(t) + D₂(t) = u (a + b)ᵀ` does not depend on `t` anywhere: it is the
    // quantity the data identifies, and it is the same map off the data.
    let summed = |amount: f64, x: &Array1<f64>| {
        let first = component_logits(&toy.write, &toy.first, &toy.direction, amount, x);
        let second = component_logits(&toy.write, &toy.second, &toy.direction, -amount, x);
        let entries: Vec<Ball> = (0..first.values.ncols())
            .map(|i| {
                Ball::new(first.values[[0, i]], first.bands[[0, i]]).add(Ball::new(second.values[[0, i]], second.bands[[0, i]]))
            })
            .collect();
        BandedMatrix {
            values: Array2::from_shape_fn((1, entries.len()), |(_, i)| entries[i].value),
            bands: Array2::from_shape_fn((1, entries.len()), |(_, i)| entries[i].radius),
        }
    };
    let reference = |_: &MemoryGovernor, _: &f64, x: &Array1<f64>| -> Result<BandedMatrix, std::fmt::Error> { Ok(summed(0.0, x)) };
    let candidate = |_: &MemoryGovernor, amount: &f64, x: &Array1<f64>| -> Result<BandedMatrix, std::fmt::Error> { Ok(summed(*amount, x)) };
    for family in [&toy.on_data, &toy.off_data] {
        let contract =
            verify_counterfactual_contract(test_governor(), &reference, &candidate, &AMOUNTS, family, &declared()).expect("verification");
        assert!(contract.certified_within(&declared()));
    }
}

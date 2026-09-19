#![cfg(test)]
//! #2946 pins for the GPT-NeoX and Qwen3 torch-layout readers: their parameters land in the blocks' fields, an absent
//! bias is zero, the output bias reaches every context's block, the absorbed gated block reproduces the layout's own
//! forward pass, and every foreign name, approximate or out-of-layout activation, missing weight and mis-shaped array
//! is refused with its typed error.

use super::{TorchLayoutError, UnabsorbedBlock, UnabsorbedGatedBlock};
use crate::response::context::{ContextBlocks, ContextDeclaredLaw};
use gam_math::gaussian_activation::{GaussianActivation, GaussianActivationError};
use gam_math::gaussian_gated::silu_derivatives;
use ndarray::{Array1, Array2, ArrayD, array};
use std::collections::BTreeMap;

fn parameters(entries: Vec<(&str, ArrayD<f64>)>) -> BTreeMap<String, ArrayD<f64>> {
    entries.into_iter().map(|(name, value)| (name.to_string(), value)).collect()
}

/// A GPT-NeoX MLP with `h = 3` units on `D = 2` inputs, writing `p = 2` outputs.
fn gpt_neox_parameters() -> BTreeMap<String, ArrayD<f64>> {
    parameters(vec![
        ("dense_h_to_4h.weight", array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn()),
        ("dense_h_to_4h.bias", array![0.5, -0.5, 1.5].into_dyn()),
        ("dense_4h_to_h.weight", array![[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]].into_dyn()),
        ("dense_4h_to_h.bias", array![0.25, -0.75].into_dyn()),
    ])
}

fn metric() -> Array2<f64> {
    array![[2.0, 0.5], [0.5, 1.0]]
}

#[test]
fn a_gpt_neox_mlp_reads_into_the_block_and_its_output_bias_reaches_each_context() {
    let block = UnabsorbedBlock::from_torch_parameters(gpt_neox_parameters(), "gelu", metric())
        .expect("a GPT-NeoX MLP with the exact GELU");
    assert_eq!(block.readers, array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    assert_eq!(block.biases, array![0.5, -0.5, 1.5]);
    assert_eq!(block.writers, array![[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]]);
    assert_eq!(block.output_bias, array![0.25, -0.75], "the block must carry dense_4h_to_h.bias");
    assert_eq!(block.metric, metric());
    assert!(matches!(block.activation, GaussianActivation::ExactGelu), "gelu is the exact GELU");

    // The zero-bias control keeps the kernel's zero-mean regime, so the block absorbs under a law whose baseline is
    // orthogonal to every reader.
    let mut unbiased = gpt_neox_parameters();
    unbiased.remove("dense_h_to_4h.bias");
    let with_output_bias =
        UnabsorbedBlock::from_torch_parameters(unbiased, "relu", metric()).expect("an MLP without reader biases");
    assert_eq!(with_output_bias.biases, Array1::<f64>::zeros(3), "an absent bias is zero");
    assert!(matches!(with_output_bias.activation, GaussianActivation::Relu));
    let contexts = ContextBlocks::new(
        with_output_bias,
        vec![ContextDeclaredLaw {
            baseline: Array1::zeros(2),
            loading: array![[1.0, 0.0], [0.0, 1.0]],
        }],
    )
    .expect("one context under the identity law");
    assert_eq!(contexts.block(0).output_bias(), array![0.25, -0.75].view());

    let mut no_output_bias = gpt_neox_parameters();
    no_output_bias.remove("dense_4h_to_h.bias");
    let block = UnabsorbedBlock::from_torch_parameters(no_output_bias, "gelu", metric()).expect("no output bias");
    assert_eq!(block.output_bias, Array1::<f64>::zeros(2));
}

#[test]
fn approximate_or_gated_activations_foreign_names_and_mis_shaped_arrays_are_refused() {
    for tag in ["gelu_new", "gelu_pytorch_tanh"] {
        let refusal = UnabsorbedBlock::from_torch_parameters(gpt_neox_parameters(), tag, metric());
        assert_eq!(
            refusal.err(),
            Some(TorchLayoutError::Activation(GaussianActivationError::ApproximateGelu {
                tag: tag.to_string(),
            })),
        );
    }
    let refusal = UnabsorbedBlock::from_torch_parameters(gpt_neox_parameters(), "tanh", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::Activation(GaussianActivationError::UnsupportedHiddenAct {
            tag: "tanh".to_string(),
        })),
    );
    let refusal = UnabsorbedBlock::from_torch_parameters(gpt_neox_parameters(), "silu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::ActivationOutsideLayout {
            hidden_act: "silu".to_string(),
        }),
    );

    let mut foreign = gpt_neox_parameters();
    foreign.insert("gate_proj.weight".to_string(), array![[1.0, 0.0]].into_dyn());
    let refusal = UnabsorbedBlock::from_torch_parameters(foreign, "gelu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::UnexpectedParameter {
            name: "gate_proj.weight".to_string(),
        }),
    );
    let mut missing = gpt_neox_parameters();
    missing.remove("dense_4h_to_h.weight");
    let refusal = UnabsorbedBlock::from_torch_parameters(missing, "gelu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::MissingParameter {
            name: "dense_4h_to_h.weight",
        }),
    );
    let mut narrow = gpt_neox_parameters();
    narrow.insert("dense_4h_to_h.weight".to_string(), array![[1.0, 0.0], [0.0, 2.0]].into_dyn());
    let refusal = UnabsorbedBlock::from_torch_parameters(narrow, "gelu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::DimensionMismatch {
            context: "dense_4h_to_h.weight columns",
            expected: 3,
            got: 2,
        }),
    );
    let mut flat = gpt_neox_parameters();
    flat.insert("dense_h_to_4h.weight".to_string(), array![1.0, 2.0, 3.0].into_dyn());
    let refusal = UnabsorbedBlock::from_torch_parameters(flat, "gelu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::WrongRank {
            name: "dense_h_to_4h.weight",
            expected: 2,
            shape: vec![3],
        }),
    );
    let mut nonfinite = gpt_neox_parameters();
    nonfinite.insert("dense_4h_to_h.bias".to_string(), array![f64::NAN, 0.0].into_dyn());
    let refusal = UnabsorbedBlock::from_torch_parameters(nonfinite, "gelu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::NonFinite {
            name: "dense_4h_to_h.bias".to_string(),
        }),
    );
    let refusal = UnabsorbedBlock::from_torch_parameters(gpt_neox_parameters(), "gelu", array![[1.0]]);
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::DimensionMismatch {
            context: "output metric rows",
            expected: 2,
            got: 1,
        }),
    );
}

/// A Qwen3 MLP with `h = 2` units on `D = 2` inputs, writing `p = 2` outputs: integers and halves.
fn qwen3_parameters() -> BTreeMap<String, ArrayD<f64>> {
    parameters(vec![
        ("gate_proj.weight", array![[1.0, -0.5], [0.5, 1.0]].into_dyn()),
        ("up_proj.weight", array![[0.0, 1.0], [1.0, 1.0]].into_dyn()),
        ("down_proj.weight", array![[1.0, -2.0], [0.5, 1.0]].into_dyn()),
    ])
}

#[test]
fn a_qwen3_mlp_reads_into_the_gated_block_and_absorbs_the_declared_law() {
    let raw = UnabsorbedGatedBlock::from_torch_parameters(qwen3_parameters(), "silu", metric())
        .expect("a Qwen3 MLP with the SiLU gate");
    assert_eq!(raw.gate_readers, array![[1.0, -0.5], [0.5, 1.0]]);
    assert_eq!(raw.up_readers, array![[0.0, 1.0], [1.0, 1.0]]);
    assert_eq!(raw.writers, array![[1.0, -2.0], [0.5, 1.0]]);
    assert_eq!(raw.gate_biases, Array1::<f64>::zeros(2), "a Qwen3 MLP holds no biases");
    assert_eq!(raw.up_biases, Array1::<f64>::zeros(2));
    assert_eq!(raw.output_bias, Array1::<f64>::zeros(2));
    assert_eq!(raw.metric, metric());

    // Absorbing h = h₀ + L z and retaining every latent coordinate must reproduce the torch layout's own forward pass
    // at h₀ + L z. The law and the point are dyadic, so both routes form the same pre-activations exactly, and they
    // differ only in the order the two output terms are added: within 2γ₂ Σ_j |u_oj t_j|.
    let baseline = array![0.5, -1.0];
    let loading = array![[1.0, 0.0], [0.5, 1.0]];
    let block = raw
        .absorb(baseline.view(), loading.view())
        .expect("the absorbed gated block");
    let latent = array![[0.5, -1.5]];
    let response = block
        .retained_response(Array2::<f64>::eye(2).view(), latent.view())
        .expect("a finite point");
    let ambient = &baseline + &loading.dot(&latent.row(0));
    let terms: Array1<f64> = (0..2)
        .map(|unit| {
            raw.up_readers.row(unit).dot(&ambient) * silu_derivatives(raw.gate_readers.row(unit).dot(&ambient))[0]
        })
        .collect();
    let executed = raw.writers.dot(&terms);
    let growth = gam_linalg::roundoff::accumulation_growth(2);
    for output in 0..2 {
        let magnitude: f64 = (0..2)
            .map(|unit| (raw.writers[[output, unit]] * terms[unit]).abs())
            .sum();
        let band = 2.0 * growth * magnitude;
        eprintln!(
            "#2946 R9 absorbed Qwen3 output {output}: operator {} torch layout {} band {band:e} quadrature {}",
            response.values[[0, output]],
            executed[output],
            response.quadrature_band[[0, output]],
        );
        assert_eq!(response.quadrature_band[[0, output]], 0.0, "retaining every coordinate smooths nothing");
        assert!((response.values[[0, output]] - executed[output]).abs() <= band);
    }
}

#[test]
fn a_qwen3_reader_refuses_a_dense_activation_foreign_names_and_mis_shaped_projections() {
    for tag in ["gelu", "relu"] {
        let refusal = UnabsorbedGatedBlock::from_torch_parameters(qwen3_parameters(), tag, metric());
        assert_eq!(
            refusal.err(),
            Some(TorchLayoutError::ActivationOutsideLayout {
                hidden_act: tag.to_string(),
            }),
        );
    }
    let mut foreign = qwen3_parameters();
    foreign.insert("dense_h_to_4h.weight".to_string(), array![[1.0, 0.0]].into_dyn());
    let refusal = UnabsorbedGatedBlock::from_torch_parameters(foreign, "silu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::UnexpectedParameter {
            name: "dense_h_to_4h.weight".to_string(),
        }),
    );
    let mut missing = qwen3_parameters();
    missing.remove("up_proj.weight");
    let refusal = UnabsorbedGatedBlock::from_torch_parameters(missing, "silu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::MissingParameter {
            name: "up_proj.weight",
        }),
    );
    let mut short = qwen3_parameters();
    short.insert("up_proj.weight".to_string(), array![[0.0, 1.0]].into_dyn());
    let refusal = UnabsorbedGatedBlock::from_torch_parameters(short, "silu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::DimensionMismatch {
            context: "up_proj.weight rows",
            expected: 2,
            got: 1,
        }),
    );
    let mut nonfinite = qwen3_parameters();
    nonfinite.insert(
        "down_proj.weight".to_string(),
        array![[f64::INFINITY, 0.0], [0.0, 1.0]].into_dyn(),
    );
    let refusal = UnabsorbedGatedBlock::from_torch_parameters(nonfinite, "silu", metric());
    assert_eq!(
        refusal.err(),
        Some(TorchLayoutError::NonFinite {
            name: "down_proj.weight".to_string(),
        }),
    );
}

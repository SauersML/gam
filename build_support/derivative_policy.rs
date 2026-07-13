//! Build-time ownership registry for production derivative specializations.
//!
//! This cohesive scanner concern owns registered production anchors, numerical
//! parity pins, retired identities, declaration discovery, and its adversarial
//! self-probes. The root build script supplies only shared filesystem, lexical,
//! and test-scope primitives.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DerivativeSpecializationKind {
    RowKernel,
    RowAtom,
    Bespoke,
}

struct DerivativeSpecialization {
    family: &'static str,
    kind: DerivativeSpecializationKind,
    production_sources: &'static [DerivativeAnchorSet],
    discovery_anchor: &'static str,
    parity_pins: &'static [DerivativeAnchorSet],
    retired_identities: &'static [&'static str],
}

struct DerivativeAnchorSet {
    path: &'static str,
    anchors: &'static [&'static str],
}

const PRODUCTION_DERIVATIVE_SPECIALIZATIONS: &[DerivativeSpecialization] = &[
    DerivativeSpecialization {
        family: "BMS rigid Bernoulli",
        kind: DerivativeSpecializationKind::RowKernel,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/bms/row_kernel.rs",
            anchors: &[
                "impl gam_math::jet_tower::RowProgram<2> for BernoulliRigidRowKernel",
                "impl RowKernel<2> for BernoulliRigidRowKernel",
            ],
        }],
        discovery_anchor: "impl RowKernel<2> for BernoulliRigidRowKernel",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/bms/gradient_paths.rs",
            anchors: &[
                "fn rigid_bernoulli_row_kernel_agrees_with_jet_tower_program_all_channels()",
                "verify_kernel_channels(&tower, &claims, 1e-9)",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "BMS FLEX Bernoulli",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/flex_row_program.rs",
                anchors: &[
                    "pub(super) struct BmsFlexRowProgram",
                    "pub(super) fn evaluate<'arena, S: RuntimeJetScalar<'arena>>(",
                    "let intercept = filtered_implicit_solve_runtime_scalar(",
                    "Ok(signed.compose_unary(self.observed_neglog_stack))",
                    "pub(super) fn try_for_each_calibration_order2<E>(",
                    "pub(super) fn try_for_each_calibration_order3_contiguous<E>(",
                    "pub(super) fn try_for_each_calibration_order4_contiguous<E>(",
                    "pub(super) fn try_for_each_order2_finalizer<E>(",
                    "pub(super) fn try_for_each_order3_finalizer<E>(",
                    "pub(super) fn try_for_each_order4_finalizer<E>(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/cell_moment_assembly.rs",
                anchors: &[
                    "pub(super) fn empirical_flex_row_third_contracted_many(",
                    "let jet = plan.evaluate(vars, 3, &workspace)?;",
                    "pub(super) fn empirical_dynamic_fourth_batch_from_plan(",
                    "let jet = plan.evaluate(vars, 4, &workspace)?;",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/row_primary_hessian.rs",
                anchors: &[
                    "pub(super) fn lower_bms_flex_row_order2(",
                    "pub(super) fn lower_bms_flex_row_order2_with_moments(",
                    "pub(super) fn lower_bms_flex_row_order2_from_parts(",
                    "BmsFlexRowProgram::try_for_each_calibration_order2(",
                    "BmsFlexRowProgram::try_for_each_calibration_order3_contiguous(",
                    "BmsFlexRowProgram::try_for_each_calibration_order4_contiguous(",
                    "BmsFlexRowProgram::try_for_each_order2_finalizer(",
                    "BmsFlexRowProgram::try_for_each_order3_finalizer(",
                    "BmsFlexRowProgram::try_for_each_order4_finalizer(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/gpu/row.rs",
                anchors: &[
                    "fn build_generated_row_kernel_source() -> String",
                    "BmsFlexRowProgram::try_for_each_calibration_order2_phase(",
                    "BmsFlexRowProgram::try_for_each_order2_finalizer_phase(",
                    "SOURCE.get_or_init(build_generated_row_kernel_source)",
                ],
            },
        ],
        discovery_anchor: "pub(super) struct BmsFlexRowProgram",
        parity_pins: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/flex_verify_932_tests.rs",
                anchors: &[
                    "fn standard_normal_flex_canonical_derivative_ladder_matches_vgh_t3_t4_932()",
                    ".lower_bms_flex_row_order2_with_moments(",
                    ".row_primary_third_contracted_with_moments(",
                    ".row_primary_fourth_contracted_ordered(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/bms/gpu/row.rs",
                anchors: &[
                    "fn generated_cuda_row_kernel_matches_canonical_cpu_lowering_415()",
                    "fn generated_cuda_row_kernel_r33_matches_canonical_cpu_lowering_932()",
                    "fn bms_flex_row_r33_consumers_match_cpu_oracles_when_cuda_available()",
                    "fn mandatory_required_gpu_workspace_consumes_device_cache_end_to_end_932()",
                    "fn release_measure_generated_bms_full_row_vs_strongest_cpu_932()",
                    "fn generated_source_interprets_compact_canonical_phase_streams()",
                ],
            },
        ],
        retired_identities: &[
            "ROW_KERNEL_BODY",
            "cpu_oracle_outputs",
            "compute_row_analytic_flex_into",
            "compute_row_analytic_flex_into_with_moments",
            "compute_row_analytic_flex_from_parts_into",
        ],
    },
    DerivativeSpecialization {
        family: "survival location-scale",
        kind: DerivativeSpecializationKind::RowKernel,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/location_scale/row_kernel.rs",
            anchors: &[
                "impl gam_math::jet_tower::RowProgram<SLS_ROW_K> for SurvivalLsRowKernel<'_>",
                "impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_>",
                "struct SlsIndexDerivativeChannels {",
                "fn project_index_diagonal<const CHANNELS: usize, const ORDER: usize>(",
                "fn lower_index_derivative_channels(self) -> SlsIndexDerivativeChannels",
                "let channels = sls_outer_plan(&kernel).lower_index_derivative_channels();",
            ],
        }],
        discovery_anchor: "impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_>",
        parity_pins: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/location_scale/tests.rs",
                anchors: &[
                    "fn survival_ls_joint_row_kernel_agrees_with_jet_tower_program_all_channels()",
                    "verify_kernel_channels(&tower, &claims, 1e-9)",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/location_scale/row_kernel.rs",
                anchors: &[
                    "fn sls_index_sparse_lowering_matches_generic_jet_all_branches_932()",
                    "fn sls_index_sparse_lowering_matches_independent_fd_all_branches_932()",
                ],
            },
        ],
        retired_identities: &["nll_index_read_channels", "SurvivalIndexNllReadChannels"],
    },
    DerivativeSpecialization {
        family: "survival marginal-slope rigid",
        kind: DerivativeSpecializationKind::RowKernel,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/marginal_slope/row_kernel.rs",
            anchors: &[
                "impl gam_math::jet_tower::RowProgram<4> for SurvivalMarginalSlopeRowKernel",
                "impl RowKernel<4> for SurvivalMarginalSlopeRowKernel",
            ],
        }],
        discovery_anchor: "impl RowKernel<4> for SurvivalMarginalSlopeRowKernel",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/marginal_slope/tests.rs",
            anchors: &[
                "fn rigid_row_kernel_agrees_with_jet_tower_program_all_channels()",
                "verify_kernel_channels(&tower, &claims, 1e-9)",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "survival marginal-slope FLEX",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/marginal_slope/timepoint_exact/flex_jet.rs",
                anchors: &[
                    "trait FlexJet: JetField + Clone {",
                    "struct FlexOuterPlan {",
                    "fn flex_row_nll<J: FlexJet>(",
                    "fn lower_flex_outer_plan_order2(",
                    "fn flex_timepoint_inputs_generic<J: FlexJet + MomentTerm>(",
                    "pub(crate) fn flex_row_nll_value_grad_hess(",
                    "pub(crate) fn flex_row_nll_third_contracted(",
                    "pub(crate) fn flex_row_nll_fourth_contracted(",
                    "pub(crate) fn compute_survival_timepoint_exact_jet(",
                    "pub(crate) fn compute_survival_timepoint_first_order_exact(",
                    "pub(crate) fn compute_survival_timepoint_directional_jet_from_cached(",
                    "pub(crate) fn compute_survival_timepoint_bidirectional_jet_from_cached(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/marginal_slope/flex_sensitivity.rs",
                anchors: &[
                    "let entry = self.compute_survival_timepoint_first_order_exact(",
                    "let (row_nll, grad, _) = self.flex_row_nll_value_grad_hess(",
                    "let entry = self.compute_survival_timepoint_exact_jet(",
                    "let (row_nll, grad, hess) = self.flex_row_nll_value_grad_hess(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-models/src/survival/marginal_slope/timepoint_exact/contracted.rs",
                anchors: &[
                    ".compute_survival_timepoint_directional_jet_from_cached(",
                    "self.flex_row_nll_third_contracted(",
                    "let entry_bi = self.compute_survival_timepoint_bidirectional_jet_from_cached(",
                    "self.flex_row_nll_fourth_contracted(",
                ],
            },
        ],
        discovery_anchor: "fn flex_row_nll<J: FlexJet>(",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/marginal_slope/timepoint_exact/flex_jet.rs",
            anchors: &[
                "fn compiled_order2_row_nll_matches_generic_plan()",
                "fn flex_timepoint_first_order_matches_jet2_and_fd_932()",
                "fn flex_timepoint_eta_chi_value_and_grad_932()",
                "fn flex_timepoint_d_cell_value_and_grad_932()",
                "fn flex_third_arena_reuses_warmed_tape_932()",
                "fn flex_timepoint_inputs_nested_dual_matches_jet4_contraction_932()",
                "fn flex_timepoint_inputs_ghw_jet4_matches_scalar_fd_932()",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "multinomial Fisher",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/multinomial_reml.rs",
            anchors: &[
                "pub struct MultinomialLogitRowProgram<'row>",
                "impl<const M: usize> gam_math::jet_tower::RowProgram<M> for MultinomialLogitRowProgram<'_>",
                "fn eval_expression<S: JetField>",
                "fn negative_log_likelihood_from_normalization",
                "pub(crate) fn value_gradient_hessian_into",
                "fn softmax_fisher_perturbation<S: FisherPerturbation>",
            ],
        }],
        discovery_anchor: "fn softmax_fisher_perturbation<S: FisherPerturbation>",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/multinomial_reml.rs",
            anchors: &[
                "fn multinomial_live_tower_matches_jet_and_fd()",
                "fn multinomial_extreme_tails_share_one_stable_row_program_932()",
            ],
        }],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "SAE reconstruction row jets",
        kind: DerivativeSpecializationKind::Bespoke,
        production_sources: &[
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/row_jet_program.rs",
                anchors: &[
                    "pub struct SaeReconstructionRowProgram {",
                    "pub fn reconstruction_all_columns_dynamic<'arena>(",
                    "pub fn beta_border_order1_dynamic<'arena>(",
                    "pub(crate) trait SaeSoftmaxRowProgramSource {",
                    "struct SoftmaxMoment<'a, S> {",
                    "pub(crate) fn execute_softmax_row_program<S: SaeSoftmaxRowProgramSource>(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/manifold/construction_row_jet_logdet_channels.rs",
                anchors: &[
                    "impl crate::row_jet_program::SaeSoftmaxRowProgramSource for ProductionSoftmaxRowProgram<'_> {",
                    "pub(crate) fn reconstruction_row_program_for_logdet(",
                    "fn fill_reconstruction_channels_from_program_dynamic(",
                    "fn fill_beta_border_channels_from_program_dynamic(",
                    "pub(crate) fn row_jets_for_logdet(",
                    "let scheduled = crate::row_jet_program::execute_softmax_row_program(",
                    "let plan = crate::gpu_kernels::sae_rowjet::plan_softmax_row_jets(",
                    "let input = crate::gpu_kernels::sae_rowjet::SaeSoftmaxRowJetInput::from_source(",
                    "let channels = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile(",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/gpu_kernels/sae_rowjet.rs",
                anchors: &[
                    "pub fn plan_softmax_row_jets(",
                    "pub fn execute_softmax_row_jet_tile(",
                    "impl SaeSoftmaxRowProgramSource for InputSource<'_> {",
                    "let scheduled = execute_softmax_row_program(&source, inv_tau, input.sqrt_row_weight);",
                    "pub const COMPLETE_SOFTMAX_KERNEL_SOURCE: &str = r#\"",
                ],
            },
        ],
        discovery_anchor: "SaeSoftmaxRowProgramSource for",
        parity_pins: &[
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/row_jet_program.rs",
                anchors: &[
                    "fn recon_jet_matches_hand_path_value_grad_hess()",
                    "fn compiled_softmax_schedule_matches_generic_tower_all_channels_932()",
                    "fn runtime_row_jets_match_fixed_oracle_above_old_arity_ceiling_932()",
                    "fn softmax_reconstruction_t3_t4_match_independent_fd_witness()",
                    "fn planted_t3_t4_corruption_is_caught_by_fd_witness()",
                    "fn planted_cross_block_sign_flip_is_caught()",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/manifold/tests_row_jet_and_outer_objective_780.rs",
                anchors: &[
                    "pub(crate) fn sae_row_jet_program_matches_production_row_jets_on_converged_cache()",
                ],
            },
            DerivativeAnchorSet {
                path: "crates/gam-sae/src/gpu_kernels/sae_rowjet.rs",
                anchors: &[
                    "fn complete_cpu_rowjet_contains_coordinate_mixed_and_beta_channels_2304()",
                    "fn memory_ledger_counts_coordinate_and_mixed_tensors_2304()",
                    "fn complete_device_matches_cpu_every_channel_when_admitted_2304()",
                ],
            },
        ],
        retired_identities: &[],
    },
    DerivativeSpecialization {
        family: "Gaussian location-scale",
        kind: DerivativeSpecializationKind::RowAtom,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/gamlss/gaussian/joint_psi.rs",
            anchors: &[
                "fn gaussian_normalized_row [generic, order2, third, fourth]",
                "pub struct GaussianJointRowProgram<'a>",
                "impl gam_math::jet_tower::RowProgram<2> for GaussianJointRowProgram<'_>",
                "pub(crate) fn row_order2(",
                "fn row_third_contracted(&self, row: usize, direction: &[f64; 2])",
                "fn row_fourth_contracted(",
                "pub(crate) fn gaussian_joint_psi_firstweights(",
                "pub(crate) fn gaussian_joint_psisecondweights(",
                "pub(crate) fn gaussian_joint_psi_mixed_driftweights(",
                "let atom = program.row_order2(i);",
                "let hessian_direction = program.row_third_contracted(i, &direction);",
                "let hessian_a_b = program.row_fourth_contracted(i, &direction_a, &direction_b);",
                "program.row_fourth_contracted(i, &drift, &psi)",
            ],
        }],
        discovery_anchor: "fn gaussian_normalized_row [generic, order2, third, fourth]",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/gamlss/gaussian/joint_psi.rs",
            anchors: &[
                "fn generated_gaussian_psi_chain_matches_generic_nested_jet_all_channels_932()",
                "fn generated_gaussian_psi_chain_matches_likelihood_finite_differences_932()",
                "fn first_directional_weights_match_jet_third()",
                "fn second_directional_weights_match_jet_fourth()",
                "crate::gamlss::GaussianJointRowProgram::new(&rows)",
            ],
        }],
        retired_identities: &[
            "cross_eta",
            "sea_seb",
            "sdea",
            "e_coef",
            "seab",
            "ma_mb",
            "de_ea",
            "kpi",
            "kdpi",
        ],
    },
    DerivativeSpecialization {
        family: "cause-specific survival",
        kind: DerivativeSpecializationKind::RowAtom,
        production_sources: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/base.rs",
            anchors: &[
                "fn cause_specific_row [generic, order2, third, fourth]",
                "pub struct CauseSpecificRowProgram",
                "impl gam_math::jet_tower::RowProgram<3> for CauseSpecificRowProgram",
            ],
        }],
        discovery_anchor: "fn cause_specific_row [generic, order2, third, fourth]",
        parity_pins: &[DerivativeAnchorSet {
            path: "crates/gam-models/src/survival/base.rs",
            anchors: &[
                "fn cause_specific_live_tower_matches_jet_and_fd()",
                "crate::survival::CauseSpecificRowProgram::new(",
            ],
        }],
        retired_identities: &[],
    },
];

#[derive(Debug)]
struct DerivativeDeclaration {
    line_index: usize,
    source: String,
}

fn normalized_rust_fragment(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, PartialEq, Eq)]
struct RustCodeToken {
    lexeme: String,
    line_index: usize,
}

fn rust_code_tokens(source: &str) -> Vec<RustCodeToken> {
    const COMPOUND_PUNCTUATION: &[&str] = &[
        "<<=", ">>=", "..=", "...", "::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=",
        "-=", "*=", "/=", "%=", "^=", "&=", "|=", "<<", ">>", "..",
    ];

    let mut tokens = Vec::new();
    for (line_index, line) in strip_file_lines(source).into_iter().enumerate() {
        let bytes = line.as_bytes();
        let mut start = 0usize;
        while start < bytes.len() {
            if bytes[start].is_ascii_whitespace() {
                start += 1;
                continue;
            }

            let end = if bytes[start].is_ascii_alphanumeric()
                || bytes[start] == b'_'
                || !bytes[start].is_ascii()
            {
                let mut end = start + 1;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric()
                        || bytes[end] == b'_'
                        || !bytes[end].is_ascii())
                {
                    end += 1;
                }
                end
            } else {
                let rest = &line[start..];
                start
                    + COMPOUND_PUNCTUATION
                        .iter()
                        .find_map(|&punctuation| {
                            rest.starts_with(punctuation).then_some(punctuation.len())
                        })
                        .unwrap_or_else(|| {
                            rest.chars()
                                .next()
                                .expect("non-empty token tail")
                                .len_utf8()
                        })
            };
            tokens.push(RustCodeToken {
                lexeme: line[start..end].to_string(),
                line_index,
            });
            start = end;
        }
    }
    tokens
}

fn code_anchor_line_indices(source: &str, anchor: &str) -> Vec<usize> {
    let source_tokens = rust_code_tokens(source);
    code_anchor_line_indices_in_tokens(&source_tokens, anchor)
}

fn code_anchor_line_indices_in_tokens(source_tokens: &[RustCodeToken], anchor: &str) -> Vec<usize> {
    let anchor_tokens = rust_code_tokens(anchor);
    if anchor_tokens.is_empty() {
        return Vec::new();
    }

    if anchor_tokens.iter().any(|token| token.lexeme == "fn") {
        let mut lines = Vec::new();
        for start in 0..source_tokens.len() {
            if source_tokens[start].lexeme != anchor_tokens[0].lexeme
                || !function_anchor_matches_at(&source_tokens[start..], &anchor_tokens)
            {
                continue;
            }
            let line_index = source_tokens[start].line_index;
            if lines.last() != Some(&line_index) {
                lines.push(line_index);
            }
        }
        return lines;
    }

    let mut lines = Vec::new();
    for window in source_tokens.windows(anchor_tokens.len()) {
        if window
            .iter()
            .zip(&anchor_tokens)
            .all(|(source, anchor)| source.lexeme == anchor.lexeme)
        {
            let line_index = window[0].line_index;
            if lines.last() != Some(&line_index) {
                lines.push(line_index);
            }
        }
    }
    lines
}

fn function_anchor_matches_at(source: &[RustCodeToken], anchor: &[RustCodeToken]) -> bool {
    let mut source_index = 0usize;
    let mut anchor_index = 0usize;
    let mut source_saw_fn = false;
    let mut anchor_saw_fn = false;
    let mut source_parameter_depth = None::<usize>;
    let mut anchor_parameter_depth = None::<usize>;
    let mut source_parameters_finished = false;
    let mut anchor_parameters_finished = false;

    while anchor_index < anchor.len() {
        if source_index >= source.len() {
            return false;
        }

        let source_has_optional_trailing_comma = source_parameter_depth == Some(1)
            && source[source_index].lexeme == ","
            && source
                .get(source_index + 1)
                .is_some_and(|token| token.lexeme == ")")
            && anchor[anchor_index].lexeme == ")";
        if source_has_optional_trailing_comma {
            source_index += 1;
            continue;
        }

        let anchor_has_optional_trailing_comma = anchor_parameter_depth == Some(1)
            && anchor[anchor_index].lexeme == ","
            && anchor
                .get(anchor_index + 1)
                .is_some_and(|token| token.lexeme == ")")
            && source[source_index].lexeme == ")";
        if anchor_has_optional_trailing_comma {
            anchor_index += 1;
            continue;
        }

        if source[source_index].lexeme != anchor[anchor_index].lexeme {
            return false;
        }

        let lexeme = source[source_index].lexeme.as_str();
        if lexeme == "fn" {
            source_saw_fn = true;
            anchor_saw_fn = true;
        } else if lexeme == "(" {
            if source_saw_fn && !source_parameters_finished {
                source_parameter_depth = Some(source_parameter_depth.map_or(1, |depth| depth + 1));
            }
            if anchor_saw_fn && !anchor_parameters_finished {
                anchor_parameter_depth = Some(anchor_parameter_depth.map_or(1, |depth| depth + 1));
            }
        } else if lexeme == ")" {
            if let Some(depth) = source_parameter_depth {
                if depth == 1 {
                    source_parameter_depth = None;
                    source_parameters_finished = true;
                } else {
                    source_parameter_depth = Some(depth - 1);
                }
            }
            if let Some(depth) = anchor_parameter_depth {
                if depth == 1 {
                    anchor_parameter_depth = None;
                    anchor_parameters_finished = true;
                } else {
                    anchor_parameter_depth = Some(depth - 1);
                }
            }
        }

        source_index += 1;
        anchor_index += 1;
    }
    true
}

fn code_identifier_line_indices(source: &str, identifier: &str) -> Vec<usize> {
    strip_file_lines(source)
        .into_iter()
        .enumerate()
        .filter_map(|(line_index, line)| {
            line.split(|character: char| character != '_' && !character.is_ascii_alphanumeric())
                .any(|token| token == identifier)
                .then_some(line_index)
        })
        .collect()
}

fn derivative_declarations(source: &str, test_mask: &[bool]) -> Vec<DerivativeDeclaration> {
    let lines = strip_file_lines(source);
    let mut declarations = Vec::new();
    let mut line_index = 0usize;
    while line_index < lines.len() {
        if test_mask.get(line_index).copied().unwrap_or(false) {
            line_index += 1;
            continue;
        }
        let trimmed = lines[line_index].trim();
        // `row_atom!` accepts normal Rust visibility (`pub`, `pub(crate)`,
        // `pub(super)`, and `pub(in path)`).  Canonicalize those declarations to
        // their `fn ...` identity before classification so visibility cannot be
        // used to hide a generated third/fourth-order specialization from this
        // registry.  Trait implementations have no visibility and stay intact.
        let canonical_function = function_declaration_without_visibility(trimmed);
        let starts_relevant_declaration = trimmed.starts_with("impl ")
            || trimmed.starts_with("impl<")
            || canonical_function.is_some();
        if !starts_relevant_declaration {
            line_index += 1;
            continue;
        }

        let start = line_index;
        let mut source = canonical_function.unwrap_or(trimmed).to_string();
        while !source.contains('{') && !source.ends_with(';') && line_index + 1 < lines.len() {
            line_index += 1;
            if test_mask.get(line_index).copied().unwrap_or(false) {
                break;
            }
            source.push(' ');
            source.push_str(lines[line_index].trim());
        }
        declarations.push(DerivativeDeclaration {
            line_index: start,
            source: normalized_rust_fragment(&source),
        });
        line_index += 1;
    }
    declarations
}

fn function_declaration_without_visibility(source: &str) -> Option<&str> {
    let mut declaration = source.trim_start();
    if let Some(after_pub) = declaration.strip_prefix("pub") {
        declaration = if let Some(after_space) = after_pub.strip_prefix(char::is_whitespace) {
            after_space.trim_start()
        } else if let Some(scoped) = after_pub.strip_prefix('(') {
            let mut depth = 1usize;
            let mut scope_end = None;
            for (index, byte) in scoped.bytes().enumerate() {
                match byte {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            scope_end = Some(index + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let after_scope = scoped.get(scope_end?..)?;
            if !after_scope.starts_with(char::is_whitespace) {
                return None;
            }
            after_scope.trim_start()
        } else {
            return None;
        };
    }
    declaration.starts_with("fn ").then_some(declaration)
}

fn implemented_trait_from_declaration(declaration: &str) -> Option<&str> {
    let Some(mut implemented) = declaration.strip_prefix("impl") else {
        return None;
    };
    implemented = implemented.trim_start();
    if implemented.starts_with('<') {
        let mut depth = 0usize;
        let mut generic_end = None;
        for (index, byte) in implemented.bytes().enumerate() {
            match byte {
                b'<' => depth += 1,
                b'>' => {
                    depth -= 1;
                    if depth == 0 {
                        generic_end = Some(index + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(generic_end) = generic_end else {
            return None;
        };
        implemented = implemented[generic_end..].trim_start();
    }
    let Some((implemented_trait, _)) = implemented.split_once(" for ") else {
        return None;
    };
    Some(implemented_trait.trim())
}

fn is_row_kernel_declaration(declaration: &str) -> bool {
    let Some(implemented_trait) = implemented_trait_from_declaration(declaration) else {
        return false;
    };
    implemented_trait.starts_with("RowKernel<") || implemented_trait.contains("::RowKernel<")
}

fn is_sae_softmax_row_program_source_declaration(declaration: &str) -> bool {
    let Some(implemented_trait) = implemented_trait_from_declaration(declaration) else {
        return false;
    };
    implemented_trait == "SaeSoftmaxRowProgramSource"
        || implemented_trait.ends_with("::SaeSoftmaxRowProgramSource")
}

fn generated_derivative_modes(declaration: &str) -> Option<(bool, bool)> {
    if !declaration.starts_with("fn ") {
        return None;
    }
    let mode_start = declaration.find('[')?;
    let mode_end = declaration[mode_start + 1..].find(']')? + mode_start + 1;
    let modes = declaration[mode_start + 1..mode_end]
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    let third = modes.contains(&"third");
    let fourth = modes.contains(&"fourth");
    (third || fourth).then_some((third, fourth))
}

fn enforce_derivative_registry_invariants() {
    for (index, specialization) in PRODUCTION_DERIVATIVE_SPECIALIZATIONS.iter().enumerate() {
        assert!(
            !specialization.production_sources.is_empty(),
            "#932 policy self-test: {} has no registered production source",
            specialization.family
        );
        assert!(
            !specialization.parity_pins.is_empty(),
            "#932 policy self-test: {} has no registered parity pin",
            specialization.family
        );
        let discovery_anchor = normalized_rust_fragment(specialization.discovery_anchor);
        assert!(
            specialization
                .production_sources
                .iter()
                .flat_map(|source| source.anchors.iter())
                .any(|anchor| normalized_rust_fragment(anchor).contains(&discovery_anchor)),
            "#932 policy self-test: {} discovery anchor is not owned by a production source",
            specialization.family
        );
        assert!(
            PRODUCTION_DERIVATIVE_SPECIALIZATIONS[index + 1..]
                .iter()
                .all(|other| other.family != specialization.family),
            "#932 policy self-test: duplicate specialization family {}",
            specialization.family
        );
    }
}

fn enforce_derivative_policy_negative_probes() {
    enforce_derivative_registry_invariants();

    for public_atom in [
        "pub fn planted_public_row [generic, third](x) { x }",
        "pub(crate) fn planted_crate_row [generic, fourth](x) { x }",
        "pub(super) fn planted_super_row [generic, third, fourth](x) { x }",
        "pub(in crate::gamlss) fn planted_scoped_row [generic, third](x) { x }",
    ] {
        let mask = compute_test_mask(public_atom, Path::new("crates/gam-models/src/planted.rs"));
        let declarations = derivative_declarations(public_atom, &mask);
        assert_eq!(
            declarations.len(),
            1,
            "#932 policy self-test: a visible generated row declaration evaded discovery: {public_atom}"
        );
        assert!(
            declarations[0].source.starts_with("fn ")
                && generated_derivative_modes(&declarations[0].source).is_some()
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::RowAtom,
                    "crates/gam-models/src/planted.rs",
                    &declarations[0].source,
                ),
            "#932 policy self-test: a visible unregistered generated row was admitted: {public_atom}"
        );
    }
    assert!(
        function_declaration_without_visibility("publication fn planted() {}").is_none(),
        "#932 policy self-test: a non-visibility `pub` prefix was parsed as a function"
    );

    let comment_only = "// impl RowKernel<7> for CommentOnlyKernel";
    assert!(
        code_anchor_line_indices(comment_only, "impl RowKernel<7> for CommentOnlyKernel")
            .is_empty(),
        "#932 policy self-test: a comment-only anchor was treated as production code"
    );

    let gaussian_order2_anchor = "fn row_order2(&self, row: usize) -> \
                                  gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 7>";
    let multiline_gaussian_order2 = "impl GaussianJointRowProgram<'_> {\n\
        pub(crate) fn row_order2(\n\
            &self,\n\
            row: usize,\n\
        ) -> gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 7> {\n\
            unreachable!()\n\
        }\n\
    }";
    assert_eq!(
        code_anchor_line_indices(multiline_gaussian_order2, gaussian_order2_anchor),
        vec![1],
        "#932 policy self-test: rustfmt line wrapping changed a production anchor's identity"
    );
    let tuple_parameter_anchor = "fn tuple_parameter(value: (usize))";
    let tuple_parameter_source = "fn tuple_parameter(value: (usize,)) {}";
    assert!(
        code_anchor_line_indices(tuple_parameter_source, tuple_parameter_anchor).is_empty(),
        "#932 policy self-test: optional parameter punctuation erased a tuple's identity"
    );

    let cfg_test_gaussian_order2 = format!("#[cfg(test)]\n{multiline_gaussian_order2}");
    let cfg_test_mask = compute_test_mask(
        &cfg_test_gaussian_order2,
        Path::new("crates/gam-models/src/planted.rs"),
    );
    let cfg_test_anchor_lines =
        code_anchor_line_indices(&cfg_test_gaussian_order2, gaussian_order2_anchor);
    assert!(
        !cfg_test_anchor_lines.is_empty()
            && cfg_test_anchor_lines
                .iter()
                .all(|line| cfg_test_mask.get(*line).copied().unwrap_or(false)),
        "#932 policy self-test: multiline anchor matching escaped cfg(test) ownership"
    );

    let gaussian_order2_near_matches = "fn row_order2_suffix(\n\
        &self, row: usize\n\
    ) -> gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 7> { unreachable!() }\n\
    fn row_order2(\n\
        &self, row: usize\n\
    ) -> gam_math::jet_scalar::StaticOrder2Atom<2, 3, 3, 70> { unreachable!() }";
    assert!(
        code_anchor_line_indices(gaussian_order2_near_matches, gaussian_order2_anchor).is_empty(),
        "#932 policy self-test: anchor matching admitted an identifier or constant suffix"
    );

    let commented_gaussian_order2 =
        format!("// {gaussian_order2_anchor}\n/* outer /* nested */ {gaussian_order2_anchor} */");
    assert!(
        code_anchor_line_indices(&commented_gaussian_order2, gaussian_order2_anchor).is_empty(),
        "#932 policy self-test: anchor matching admitted line or block comments"
    );

    let row_kernel = "impl RowKernel<7> for PlantedKernel {}";
    let row_mask = compute_test_mask(row_kernel, Path::new("crates/gam-models/src/planted.rs"));
    let row_declarations = derivative_declarations(row_kernel, &row_mask);
    assert!(
        row_declarations.iter().any(|declaration| {
            is_row_kernel_declaration(&declaration.source)
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::RowKernel,
                    "crates/gam-models/src/planted.rs",
                    &declaration.source,
                )
        }),
        "#932 policy self-test: an unregistered RowKernel was not discovered"
    );

    let registered_row_path = "crates/gam-models/src/survival/location_scale/row_kernel.rs";
    let registered_row_source =
        "impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_> {";
    let registered_row_mask =
        compute_test_mask(registered_row_source, Path::new(registered_row_path));
    assert!(
        derivative_declarations(registered_row_source, &registered_row_mask)
            .iter()
            .any(|declaration| {
                is_row_kernel_declaration(&declaration.source)
                    && specialization_site_is_registered(
                        DerivativeSpecializationKind::RowKernel,
                        registered_row_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: the exact registered survival RowKernel was not admitted"
    );
    let same_file_rogue_row =
        "impl crate::row_kernel::RowKernel<SLS_ROW_K> for PlantedSameFileKernel {";
    let same_file_rogue_row_mask =
        compute_test_mask(same_file_rogue_row, Path::new(registered_row_path));
    assert!(
        derivative_declarations(same_file_rogue_row, &same_file_rogue_row_mask)
            .iter()
            .any(|declaration| {
                is_row_kernel_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowKernel,
                        registered_row_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: a rogue RowKernel in the registered survival source was admitted"
    );

    let bounded_helper =
        "impl<const K: usize, T: RowKernel<K>> HyperOperator for PlantedWrapper<K, T> {}";
    let bounded_mask = compute_test_mask(
        bounded_helper,
        Path::new("crates/gam-models/src/planted.rs"),
    );
    assert!(
        derivative_declarations(bounded_helper, &bounded_mask)
            .iter()
            .all(|declaration| !is_row_kernel_declaration(&declaration.source)),
        "#932 policy self-test: a generic RowKernel bound was mistaken for a RowKernel implementation"
    );

    let separate_generated = "row_atom! {\n    fn planted_third [generic, third](x) { x }\n    fn planted_fourth [generic, fourth](x) { x }\n}";
    let generated_mask = compute_test_mask(
        separate_generated,
        Path::new("crates/gam-models/src/planted.rs"),
    );
    let generated_declarations = derivative_declarations(separate_generated, &generated_mask);
    let unregistered_generated = generated_declarations
        .iter()
        .filter(|declaration| {
            generated_derivative_modes(&declaration.source).is_some()
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::RowAtom,
                    "crates/gam-models/src/planted.rs",
                    &declaration.source,
                )
        })
        .count();
    assert_eq!(
        unregistered_generated, 2,
        "#932 policy self-test: separate generated-third/fourth declarations were not both discovered"
    );

    let registered_gaussian_path = "crates/gam-models/src/gamlss/gaussian/joint_psi.rs";
    let registered_gaussian_atom =
        "fn gaussian_normalized_row [generic, order2, third, fourth](delta_mu) { delta_mu }";
    let registered_gaussian_mask = compute_test_mask(
        registered_gaussian_atom,
        Path::new(registered_gaussian_path),
    );
    assert!(
        derivative_declarations(registered_gaussian_atom, &registered_gaussian_mask)
            .iter()
            .any(|declaration| {
                generated_derivative_modes(&declaration.source).is_some()
                    && specialization_site_is_registered(
                        DerivativeSpecializationKind::RowAtom,
                        registered_gaussian_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: the exact registered Gaussian row atom was not admitted"
    );
    let same_file_rogue_atom = "fn planted_gaussian_row [generic, order2, third, fourth](x) { x }";
    let same_file_rogue_atom_mask =
        compute_test_mask(same_file_rogue_atom, Path::new(registered_gaussian_path));
    assert!(
        derivative_declarations(same_file_rogue_atom, &same_file_rogue_atom_mask)
            .iter()
            .any(|declaration| {
                generated_derivative_modes(&declaration.source).is_some()
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowAtom,
                        registered_gaussian_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: a rogue row atom in the registered Gaussian source was admitted"
    );

    let sae_source = "impl SaeSoftmaxRowProgramSource for PlantedSaeSource {}";
    let sae_mask = compute_test_mask(sae_source, Path::new("crates/gam-sae/src/planted.rs"));
    let sae_declarations = derivative_declarations(sae_source, &sae_mask);
    assert!(
        sae_declarations.iter().any(|declaration| {
            is_sae_softmax_row_program_source_declaration(&declaration.source)
                && !specialization_site_is_registered(
                    DerivativeSpecializationKind::Bespoke,
                    "crates/gam-sae/src/planted.rs",
                    &declaration.source,
                )
        }),
        "#932 policy self-test: an unregistered SAE softmax row-program source was not discovered"
    );
    let sae_bound = "fn planted<S: SaeSoftmaxRowProgramSource>(source: &S) {}";
    let sae_bound_mask = compute_test_mask(sae_bound, Path::new("crates/gam-sae/src/planted.rs"));
    assert!(
        derivative_declarations(sae_bound, &sae_bound_mask)
            .iter()
            .all(|declaration| {
                !is_sae_softmax_row_program_source_declaration(&declaration.source)
            }),
        "#932 policy self-test: an SAE source bound was mistaken for a trait implementation"
    );

    let registered_sae_path = "crates/gam-sae/src/gpu_kernels/sae_rowjet.rs";
    let registered_sae_source = "impl SaeSoftmaxRowProgramSource for InputSource<'_> {";
    let registered_sae_mask =
        compute_test_mask(registered_sae_source, Path::new(registered_sae_path));
    assert!(
        derivative_declarations(registered_sae_source, &registered_sae_mask)
            .iter()
            .any(|declaration| {
                is_sae_softmax_row_program_source_declaration(&declaration.source)
                    && specialization_site_is_registered(
                        DerivativeSpecializationKind::Bespoke,
                        registered_sae_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: the exact registered SAE implementation was not admitted"
    );
    let same_file_rogue = "impl SaeSoftmaxRowProgramSource for PlantedSameFileSource {}";
    let same_file_rogue_mask = compute_test_mask(same_file_rogue, Path::new(registered_sae_path));
    assert!(
        derivative_declarations(same_file_rogue, &same_file_rogue_mask)
            .iter()
            .any(|declaration| {
                is_sae_softmax_row_program_source_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::Bespoke,
                        registered_sae_path,
                        &declaration.source,
                    )
            }),
        "#932 policy self-test: a rogue SAE implementation in a registered source file was admitted"
    );

    let retired = "fn planted_retired_identity() {}";
    assert_eq!(
        code_identifier_line_indices(retired, "planted_retired_identity"),
        vec![0],
        "#932 policy self-test: a retired derivative identity was not discovered"
    );
    assert!(
        code_identifier_line_indices(
            "// planted_retired_identity\nfn planted_retired_identity_suffix() {}",
            "planted_retired_identity",
        )
        .is_empty(),
        "#932 policy self-test: retired-identity matching ignored token boundaries or comments"
    );
}

pub(super) fn enforce_production_derivative_specializations(root: &Path) {
    enforce_derivative_policy_negative_probes();
    let mut violations = Vec::new();
    for specialization in PRODUCTION_DERIVATIVE_SPECIALIZATIONS {
        for production in specialization.production_sources {
            let production_path = root.join(production.path);
            match fs::read_to_string(&production_path) {
                Ok(source) => {
                    let test_mask = compute_test_mask(&source, Path::new(production.path));
                    let source_tokens = rust_code_tokens(&source);
                    for anchor in production.anchors {
                        let anchor_lines =
                            code_anchor_line_indices_in_tokens(&source_tokens, anchor);
                        if anchor_lines.is_empty() {
                            violations.push(format!(
                                "{} production anchor is missing from {}: {}",
                                specialization.family, production.path, anchor
                            ));
                        } else if anchor_lines
                            .iter()
                            .all(|line| test_mask.get(*line).copied().unwrap_or(false))
                        {
                            violations.push(format!(
                                "{} production anchor is gated by cfg(test) in {}: {}",
                                specialization.family, production.path, anchor
                            ));
                        }
                    }
                }
                Err(error) => violations.push(format!(
                    "{} production source {} cannot be read: {error}",
                    specialization.family, production.path
                )),
            }
        }

        for pin in specialization.parity_pins {
            let pin_path = root.join(pin.path);
            match fs::read_to_string(&pin_path) {
                Ok(source) => {
                    let source_tokens = rust_code_tokens(&source);
                    for anchor in pin.anchors {
                        if code_anchor_line_indices_in_tokens(&source_tokens, anchor).is_empty() {
                            violations.push(format!(
                                "{} registered parity pin is missing from {}: {}",
                                specialization.family, pin.path, anchor
                            ));
                        }
                    }
                }
                Err(error) => violations.push(format!(
                    "{} pin source {} cannot be read: {error}",
                    specialization.family, pin.path
                )),
            }
        }
    }

    visit_files(
        root,
        &root.join("crates/gam-models/src"),
        &mut |rel, content| {
            let rel_path = rel.to_string_lossy().replace('\\', "/");
            if rel.extension().and_then(OsStr::to_str) != Some("rs") {
                return;
            }
            for specialization in PRODUCTION_DERIVATIVE_SPECIALIZATIONS {
                for identifier in specialization.retired_identities {
                    for line_index in code_identifier_line_indices(content, identifier) {
                        violations.push(format!(
                            "{} retired derivative identity reappeared at {rel_path}:{}: {}",
                            specialization.family,
                            line_index + 1,
                            identifier
                        ));
                    }
                }
            }
            let test_mask = compute_test_mask(content, rel);
            for declaration in derivative_declarations(content, &test_mask) {
                if is_row_kernel_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowKernel,
                        &rel_path,
                        &declaration.source,
                    )
                {
                    violations.push(format!(
                        "unregistered production RowKernel specialization at {rel_path}:{}: {}",
                        declaration.line_index + 1,
                        declaration.source
                    ));
                }

                if generated_derivative_modes(&declaration.source).is_some()
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::RowAtom,
                        &rel_path,
                        &declaration.source,
                    )
                {
                    violations.push(format!(
                        "unregistered generated third/fourth row specialization at {rel_path}:{}: {}",
                        declaration.line_index + 1,
                        declaration.source
                    ));
                }
            }
        },
    );

    visit_files(
        root,
        &root.join("crates/gam-sae/src"),
        &mut |rel, content| {
            let rel_path = rel.to_string_lossy().replace('\\', "/");
            if rel.extension().and_then(OsStr::to_str) != Some("rs") {
                return;
            }
            let test_mask = compute_test_mask(content, rel);
            for declaration in derivative_declarations(content, &test_mask) {
                if is_sae_softmax_row_program_source_declaration(&declaration.source)
                    && !specialization_site_is_registered(
                        DerivativeSpecializationKind::Bespoke,
                        &rel_path,
                        &declaration.source,
                    )
                {
                    violations.push(format!(
                        "unregistered production SAE softmax row-program source at {rel_path}:{}: {}",
                        declaration.line_index + 1,
                        declaration.source
                    ));
                }
            }
        },
    );

    if violations.is_empty() {
        return;
    }
    for violation in &violations {
        println!("cargo:warning=#932 derivative-specialization policy: {violation}");
    }
    panic!(
        "#932 derivative-specialization registry rejected {} violation(s)",
        violations.len()
    );
}

fn specialization_site_is_registered(
    kind: DerivativeSpecializationKind,
    path: &str,
    source_line: &str,
) -> bool {
    let normalized_source = normalized_rust_fragment(source_line);
    PRODUCTION_DERIVATIVE_SPECIALIZATIONS
        .iter()
        .any(|specialization| {
            specialization.kind == kind
                && specialization
                    .production_sources
                    .iter()
                    .filter(|source| source.path == path)
                    .flat_map(|source| source.anchors.iter())
                    .any(|anchor| registered_declaration_matches_anchor(&normalized_source, anchor))
        })
}

fn registered_declaration_matches_anchor(declaration: &str, anchor: &str) -> bool {
    let anchor = normalized_rust_fragment(anchor);
    if !(anchor.starts_with("impl ") || anchor.starts_with("impl<") || anchor.starts_with("fn ")) {
        return false;
    }
    let Some(remainder) = declaration.strip_prefix(&anchor) else {
        return false;
    };
    if remainder.is_empty() {
        return true;
    }
    if anchor.ends_with('(') || remainder.starts_with('(') {
        return true;
    }
    if !remainder.chars().next().is_some_and(char::is_whitespace) {
        return false;
    }
    let declaration_tail = remainder.trim_start();
    declaration_tail.starts_with('{') || declaration_tail.starts_with("where ")
}

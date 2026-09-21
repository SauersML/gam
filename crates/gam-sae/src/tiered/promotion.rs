//! The code-space census's accepted single-block promotions installed as certified
//! curved charts (#2023).
//!
//! The census ([`super::code_space`]) prices, in bits, whether ONE curved chart describes
//! a Tier-1 block's firings more cheaply than the block's `b` linear atoms, and F23's
//! acceptance ([`crate::manifold::curve_promotion::propose_curve_promotion`]) admits a
//! verdict only when the ring is recognized and the atomic ledger strictly buys it. A
//! verdict mutates nothing. This module makes an accepted verdict a model move: the
//! block's code cloud (exactly the firings the ledger priced, read off
//! [`CodeSpacePromotionReport::proposal_firings`]) is fit by the census's own certified
//! chart fitter, [`fit_pair_chart`], and the block's firings decode through that chart
//! instead of through their free amplitudes.
//!
//! The chart's smoothing is selected by REML, its inner fixed point stops on the derived
//! tolerance, and its outer stationarity certificate travels with the fit, so an installed
//! chart carries a certificate and no caller-set budget, step size or ridge. A fit that does
//! not certify, or a cloud the fitter cannot chart, is a refusal carrying its own reason.
//!
//! A certified chart must also deliver the bits the census priced. The ledger priced the
//! curved arm as the least circle phase code meeting the flat arm's in-plane distortion
//! `D`, with the ring's radial spread `Var(r)` as its fitting error. The installed chart's
//! fitting error is its own reconstruction error per firing, so the delivered code is the
//! least phase code meeting `D` with that error in place of `Var(r)`. When that code needs
//! more codewords than the priced one, or no finite codebook meets `D`, the census's saved
//! bits are not delivered and the promotion is refused by name.
//!
//! Only single-block communities install. A cross-block pair chart decodes only the rows
//! where both blocks fire, so installing one needs a per-row partition of the routed
//! slots among overlapping communities, which is a different move.

use std::fmt;

use ndarray::Array2;

use crate::description_length::{CirclePhaseCode, circle_phase_code};
use crate::manifold::curve_promotion::CurvePromotionProposal;
use crate::sparse_dict::BlockSparseFit;
use crate::tiered::code_space::{CodeSpacePromotionReport, PairChartFit, fit_pair_chart};

/// One accepted census promotion installed as a certified curved chart.
#[derive(Clone, Debug)]
pub struct InstalledChart {
    /// The Tier-1 block whose linear atoms the chart replaces.
    pub block: usize,
    /// Routed `(row, slot)` of each firing, in the order the census built the cloud.
    pub firings: Vec<(usize, usize)>,
    /// The certified chart fit.
    pub fit: PairChartFit,
    /// The bits the census priced the promotion as saving, `dl_old − dl_new`.
    pub dl_saved_bits: f64,
}

/// An accepted promotion whose chart did not install, with the reason.
#[derive(Clone, Debug)]
pub struct PromotionRefusal {
    /// The Tier-1 block the promotion would have replaced.
    pub block: usize,
    /// The bits the census priced the promotion as saving.
    pub dl_saved_bits: f64,
    /// Why the chart did not install.
    pub reason: PromotionRefusalReason,
}

/// Why an accepted promotion's chart did not install.
#[derive(Clone, Debug, PartialEq)]
pub enum PromotionRefusalReason {
    /// The chart fitter refused the cloud; its reason verbatim.
    ChartFit(String),
    /// The chart fit ran, but its outer certificate or its inner fixed point failed.
    NotCertified { certified: bool, recurred: bool },
    /// The certified chart reconstructs the cloud less faithfully than the priced code:
    /// with the chart's error per firing in place of the ring's radial spread, the least
    /// phase code meeting the flat arm's distortion needs `delivered_codebook` codewords
    /// (`None`: no finite codebook) against the `priced_codebook` the census priced.
    FidelityBelowPriced {
        chart_distortion: f64,
        flat_distortion: f64,
        priced_codebook: u64,
        delivered_codebook: Option<u64>,
    },
}

impl fmt::Display for PromotionRefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChartFit(reason) => write!(f, "the chart fitter refused the cloud: {reason}"),
            Self::NotCertified {
                certified,
                recurred,
            } => write!(
                f,
                "the chart fit did not certify (outer certificate {certified}, inner fixed \
                 point recurred {recurred})"
            ),
            Self::FidelityBelowPriced {
                chart_distortion,
                flat_distortion,
                priced_codebook,
                delivered_codebook: Some(delivered),
            } => write!(
                f,
                "the certified chart's distortion {chart_distortion:.6e} per firing leaves the flat \
                 arm's distortion {flat_distortion:.6e} needing {delivered} phase codewords, against \
                 the {priced_codebook} the census priced"
            ),
            Self::FidelityBelowPriced {
                chart_distortion,
                flat_distortion,
                priced_codebook,
                delivered_codebook: None,
            } => write!(
                f,
                "the certified chart's distortion {chart_distortion:.6e} per firing leaves no finite \
                 phase codebook meeting the flat arm's distortion {flat_distortion:.6e}, which the \
                 census priced at {priced_codebook} codewords"
            ),
        }
    }
}

/// The fidelity refusal of a certified chart whose distortion per firing is
/// `chart_distortion` (module note), or `None` when the chart delivers the `priced` code:
/// the least phase code meeting `flat_distortion` on a ring of mean radius `mean_radius`,
/// with the chart's error as the fitting error, needs no more codewords than `priced`.
fn fidelity_refusal(
    priced: &CirclePhaseCode,
    mean_radius: f64,
    flat_distortion: f64,
    chart_distortion: f64,
) -> Result<Option<PromotionRefusalReason>, String> {
    let delivered = circle_phase_code(mean_radius, chart_distortion, flat_distortion)?;
    Ok(match delivered {
        Some(code) if code.codebook_size <= priced.codebook_size => None,
        delivered => Some(PromotionRefusalReason::FidelityBelowPriced {
            chart_distortion,
            flat_distortion,
            priced_codebook: priced.codebook_size,
            delivered_codebook: delivered.map(|code| code.codebook_size),
        }),
    })
}

/// The priced phase code of an accepted proposal. The ledger accepts only a finite
/// `DL_new`, which needs a phase code, so its absence is an error, not a refusal.
fn priced_phase_code(proposal: &CurvePromotionProposal) -> Result<CirclePhaseCode, String> {
    proposal.curved_phase_code.ok_or_else(|| {
        format!(
            "install_block_promotions: block {} was accepted with no priced phase code",
            proposal.block
        )
    })
}

/// Every accepted single-block promotion of one census, installed or refused.
#[derive(Clone, Debug, Default)]
pub struct PromotionInstall {
    /// The installed charts, in census order.
    pub charts: Vec<InstalledChart>,
    /// The accepted promotions whose chart did not install, in census order.
    pub refusals: Vec<PromotionRefusal>,
}

/// Install every accepted single-block census promotion as a certified curved chart on
/// the Tier-1 routing the census read.
pub(crate) fn install_block_promotions(
    census: &CodeSpacePromotionReport,
    tier1: &BlockSparseFit,
) -> Result<PromotionInstall, String> {
    if census.proposal_firings.len() != census.proposals.len() {
        return Err(format!(
            "install_block_promotions: {} firing lists for {} proposals",
            census.proposal_firings.len(),
            census.proposals.len()
        ));
    }
    let b = tier1.block_size;
    let mut install = PromotionInstall::default();
    for (proposal, firings) in census.proposals.iter().zip(&census.proposal_firings) {
        if !proposal.accept {
            continue;
        }
        let dl_saved_bits = proposal.dl_old - proposal.dl_new;
        let mut cloud = Array2::<f64>::zeros((firings.len(), b));
        for (firing, &(row, slot)) in firings.iter().enumerate() {
            for axis in 0..b {
                cloud[[firing, axis]] = tier1.codes[[row, slot, axis]] as f64;
            }
        }
        let refusal = match fit_pair_chart(cloud.view(), proposal.block as u64) {
            Ok(fit) if fit.certified && fit.recurred => {
                let priced = priced_phase_code(proposal)?;
                let refusal = fidelity_refusal(
                    &priced,
                    proposal.mean_radius,
                    proposal.flat_distortion,
                    fit.distortion,
                )?;
                let Some(reason) = refusal else {
                    install.charts.push(InstalledChart {
                        block: proposal.block,
                        firings: firings.clone(),
                        fit,
                        dl_saved_bits,
                    });
                    continue;
                };
                reason
            }
            Ok(fit) => PromotionRefusalReason::NotCertified {
                certified: fit.certified,
                recurred: fit.recurred,
            },
            Err(reason) => PromotionRefusalReason::ChartFit(reason),
        };
        install.refusals.push(PromotionRefusal {
            block: proposal.block,
            dl_saved_bits,
            reason: refusal,
        });
    }
    Ok(install)
}

#[cfg(test)]
mod promotion_tests {
    use super::*;

    /// #2023 fidelity refusal, on the priced code of a ring with mean radius 1, radial
    /// variance `1e-4` and flat distortion `2e-3`. A chart whose error per firing is that
    /// radial variance, or zero, delivers the priced code. A chart whose error sits between
    /// the variance and the distortion leaves less of the budget for the phase index, so
    /// the delivered code needs more codewords and is refused with its count. A chart whose
    /// error alone spends the whole distortion leaves no finite codebook and is refused
    /// with none.
    #[test]
    fn a_chart_is_refused_exactly_when_it_needs_more_phase_codewords_than_priced_2023() {
        let (mean_radius, radial_variance, flat_distortion) = (1.0, 1.0e-4, 2.0e-3);
        let priced = circle_phase_code(mean_radius, radial_variance, flat_distortion)
            .expect("finite inputs")
            .expect("the budget exceeds the radial variance");
        for chart_distortion in [radial_variance, 0.0] {
            assert_eq!(
                fidelity_refusal(&priced, mean_radius, flat_distortion, chart_distortion)
                    .expect("finite inputs"),
                None,
                "a chart at or below the ring's radial spread delivers the priced code"
            );
        }
        let between = 0.5 * (radial_variance + flat_distortion);
        let delivered = circle_phase_code(mean_radius, between, flat_distortion)
            .expect("finite inputs")
            .expect("the budget exceeds the chart's error");
        assert!(
            delivered.codebook_size > priced.codebook_size,
            "the control must need more codewords: delivered {} against priced {}",
            delivered.codebook_size,
            priced.codebook_size
        );
        assert_eq!(
            fidelity_refusal(&priced, mean_radius, flat_distortion, between).expect("finite inputs"),
            Some(PromotionRefusalReason::FidelityBelowPriced {
                chart_distortion: between,
                flat_distortion,
                priced_codebook: priced.codebook_size,
                delivered_codebook: Some(delivered.codebook_size),
            })
        );
        assert_eq!(
            fidelity_refusal(&priced, mean_radius, flat_distortion, flat_distortion)
                .expect("finite inputs"),
            Some(PromotionRefusalReason::FidelityBelowPriced {
                chart_distortion: flat_distortion,
                flat_distortion,
                priced_codebook: priced.codebook_size,
                delivered_codebook: None,
            })
        );
    }
}

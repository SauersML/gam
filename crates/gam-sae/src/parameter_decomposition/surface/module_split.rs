//! `module_split`: the merged MLP normal form, its finest additive blocks and optimal
//! splits ([`crate::parameter_decomposition::module_split`]) on the wire.

use std::collections::BTreeMap;

use gam_math::gaussian_activation::GaussianActivation;
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayD};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::state_quotient::{SpectralNormBoundsReport, bounds_report};
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, matrix, output, vector};
use crate::parameter_decomposition::module_split::{
    AdditiveBlocks, MlpNormalForm, OptimalSplit, SplitDomain, UnitSource,
};

/// A block `F(h) = W_out σ(W_in h + b_in) + b_out + L h`. Every string is the id of an
/// input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MlpBlockRequest {
    /// The Hugging Face activation tag (`gelu` is the exact GELU, or `relu`).
    pub hidden_act: String,
    pub w_in: String,
    pub b_in: String,
    pub w_out: String,
    pub b_out: String,
    /// The linear skip `L`, or null for none.
    pub skip: Option<String>,
}

/// The block, and what to compute on its normal form.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleSplitRequest {
    pub block: MlpBlockRequest,
    /// Proposed merged-unit subsets, each split optimally against its complement
    /// (`MlpNormalForm::optimal_split`).
    pub subsets: Vec<Vec<usize>>,
    /// Ids of vectors `q` (one entry per merged unit) to apply the pair-weight
    /// Laplacian to (`AdditiveBlocks::laplacian_apply`).
    pub laplacian_vectors: Vec<String>,
}

/// The normal form, the blocks, the requested splits and Laplacian products.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModuleSplitReport {
    pub normal_form: NormalFormReport,
    pub blocks: AdditiveBlocksReport,
    pub splits: Vec<OptimalSplitReport>,
    /// Ids of `𝓛 q`, one per requested vector.
    pub laplacian: Vec<String>,
}

/// [`MlpNormalForm`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormalFormReport {
    /// Ids of `L`, `c`, the merged reads, biases and writes.
    pub linear: String,
    pub linear_band: f64,
    pub offset: String,
    pub reads: String,
    pub biases: String,
    pub writes: String,
    pub write_bands: Vec<f64>,
    pub sources: Vec<Vec<UnitSourceReport>>,
    pub cancelled: Vec<usize>,
    pub constant: Vec<usize>,
    pub lipschitz: f64,
}

/// [`UnitSource`] on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct UnitSourceReport {
    pub unit: usize,
    pub negated: bool,
}

/// [`SplitDomain`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SplitDomainReport {
    Reads { units: usize },
    UnitBall { input_dimension: usize },
}

pub type SplitStatusWire = EvidenceStatusWire<(), SplitDomainReport>;

/// [`AdditiveBlocks`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdditiveBlocksReport {
    /// Ids of `U` (`m × r`) and `T` (`r × d_in`).
    pub frame: String,
    pub coordinates: String,
    pub singular_values: Vec<f64>,
    pub rank: SplitStatusWire,
    pub projector_band: f64,
    pub free_input_dimension: usize,
    pub certified_pairs: usize,
    pub unresolved_pairs: usize,
    pub finest: Vec<Vec<usize>>,
    pub unresolved_joins: Vec<UnresolvedJoinReport>,
    pub components: Vec<OptimalSplitReport>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct UnresolvedJoinReport {
    pub left: usize,
    pub right: usize,
    pub pairs: usize,
    pub largest: f64,
    pub largest_band: f64,
}

/// [`OptimalSplit`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OptimalSplitReport {
    pub subset: Vec<usize>,
    pub eigenvalues: Vec<f64>,
    pub loss: SplitStatusWire,
    pub cut: SplitStatusWire,
    /// Ids of `P`'s orthonormal rows and of `Ŵ`.
    pub projector_basis: String,
    pub approximate_reads: String,
    pub read_distance: SpectralNormBoundsReport,
    /// Per unit radius; scale by the declared radius.
    pub native_bound: SplitStatusWire,
    pub cut_bound: f64,
}

fn split_domain(domain: SplitDomain) -> SplitDomainReport {
    match domain {
        SplitDomain::Reads { units } => SplitDomainReport::Reads { units },
        SplitDomain::UnitBall { input_dimension } => SplitDomainReport::UnitBall { input_dimension },
    }
}

fn status(
    status: crate::parameter_decomposition::supports::EvidenceStatus<(), SplitDomain>,
) -> Result<SplitStatusWire, MpdSurfaceError> {
    EvidenceStatusWire::from_status_with(status, |witness| witness, split_domain)
}

fn finite_list(field: &'static str, values: &[f64]) -> Result<Vec<f64>, MpdSurfaceError> {
    values.iter().map(|&value| finite(field, value)).collect()
}

/// The owner's normal form of a wire block.
pub(super) fn normal_form(
    tensors: &BTreeMap<String, ArrayD<f64>>,
    block: &MlpBlockRequest,
) -> Result<MlpNormalForm, MpdSurfaceError> {
    let activation = GaussianActivation::from_hidden_act(&block.hidden_act)
        .map_err(|error| MpdSurfaceError::InvalidRequest(format!("{error:?}")))?;
    let skip = match &block.skip {
        Some(id) => Some(matrix(tensors, id)?),
        None => None,
    };
    MlpNormalForm::new(
        activation,
        matrix(tensors, &block.w_in)?,
        vector(tensors, &block.b_in)?,
        matrix(tensors, &block.w_out)?,
        vector(tensors, &block.b_out)?,
        skip,
    )
    .map_err(|error| MpdSurfaceError::ModuleSplit(error.to_string()))
}

struct Arrays(BTreeMap<String, ArrayD<f64>>);

impl Arrays {
    fn put2(&mut self, id: String, array: Array2<f64>) -> String {
        self.0.insert(id.clone(), array.into_dyn());
        id
    }

    fn put1(&mut self, id: String, array: Array1<f64>) -> String {
        self.0.insert(id.clone(), array.into_dyn());
        id
    }

    fn split(&mut self, prefix: String, split: OptimalSplit) -> Result<OptimalSplitReport, MpdSurfaceError> {
        Ok(OptimalSplitReport {
            eigenvalues: finite_list("eigenvalues", &split.eigenvalues)?,
            loss: status(split.loss)?,
            cut: status(split.cut)?,
            projector_basis: self.put2(format!("{prefix}/projector_basis"), split.projector_basis),
            approximate_reads: self.put2(format!("{prefix}/approximate_reads"), split.approximate_reads),
            read_distance: bounds_report("read_distance", split.read_distance)?,
            native_bound: status(split.native_bound)?,
            cut_bound: finite("cut_bound", split.cut_bound)?,
            subset: split.subset,
        })
    }
}

pub(super) fn run(
    request: ModuleSplitRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let error = |error: crate::parameter_decomposition::module_split::ModuleSplitError| {
        MpdSurfaceError::ModuleSplit(error.to_string())
    };
    let form = normal_form(tensors, &request.block)?;
    let blocks = form.additive_blocks(governor).map_err(error)?;
    let splits = request
        .subsets
        .iter()
        .map(|subset| form.optimal_split(governor, &blocks, subset).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    let laplacian = request
        .laplacian_vectors
        .iter()
        .map(|id| blocks.laplacian_apply(vector(tensors, id)?).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    project(form, blocks, splits, laplacian)
}

pub(super) fn project(
    form: MlpNormalForm,
    blocks: AdditiveBlocks,
    splits: Vec<OptimalSplit>,
    laplacian: Vec<Array1<f64>>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut arrays = Arrays(BTreeMap::new());
    let normal_form = NormalFormReport {
        linear_band: finite("linear_band", form.linear_band)?,
        write_bands: finite_list("write_bands", &form.write_bands)?,
        lipschitz: finite("lipschitz", form.lipschitz)?,
        sources: form
            .sources
            .iter()
            .map(|sources| {
                sources
                    .iter()
                    .map(|&UnitSource { unit, negated }| UnitSourceReport { unit, negated })
                    .collect()
            })
            .collect(),
        cancelled: form.cancelled.clone(),
        constant: form.constant.clone(),
        linear: arrays.put2("normal_form/linear".to_string(), form.linear),
        offset: arrays.put1("normal_form/offset".to_string(), form.offset),
        reads: arrays.put2("normal_form/reads".to_string(), form.reads),
        biases: arrays.put1("normal_form/biases".to_string(), form.biases),
        writes: arrays.put2("normal_form/writes".to_string(), form.writes),
    };
    let components = blocks
        .components
        .into_iter()
        .enumerate()
        .map(|(index, split)| arrays.split(format!("blocks/components/{index}"), split))
        .collect::<Result<Vec<_>, _>>()?;
    let blocks_report = AdditiveBlocksReport {
        singular_values: finite_list("singular_values", &blocks.singular_values)?,
        rank: status(blocks.rank)?,
        projector_band: finite("projector_band", blocks.projector_band)?,
        free_input_dimension: blocks.free_input_dimension,
        certified_pairs: blocks.certified_pairs,
        unresolved_pairs: blocks.unresolved_pairs,
        finest: blocks.finest,
        unresolved_joins: blocks
            .unresolved_joins
            .iter()
            .map(|join| UnresolvedJoinReport {
                left: join.left,
                right: join.right,
                pairs: join.pairs,
                largest: join.largest,
                largest_band: join.largest_band,
            })
            .collect(),
        components,
        frame: arrays.put2("blocks/frame".to_string(), blocks.frame),
        coordinates: arrays.put2("blocks/coordinates".to_string(), blocks.coordinates),
    };
    let splits = splits
        .into_iter()
        .enumerate()
        .map(|(index, split)| arrays.split(format!("splits/{index}"), split))
        .collect::<Result<Vec<_>, _>>()?;
    let laplacian = laplacian
        .into_iter()
        .enumerate()
        .map(|(index, product)| arrays.put1(format!("laplacian/{index}"), product))
        .collect();
    Ok(output(
        MpdResult::ModuleSplit(Box::new(ModuleSplitReport {
            normal_form,
            blocks: blocks_report,
            splits,
            laplacian,
        })),
        arrays.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::parameter_decomposition::test_support::test_governor;
    use ndarray::array;

    /// Two planted modules on disjoint input coordinates (units 0-1 read h₀, h₁; unit 2
    /// reads h₂), plus a unit and its exact negation, which merge.
    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        let w_in = array![
            [1.0, 0.5, 0.0],
            [-0.5, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0]
        ];
        let b_in = array![0.1, -0.2, 0.3, 0.5, -0.5];
        let w_out = array![[1.0, 0.0, 0.5, 1.0, 1.0], [0.0, 2.0, -1.0, 0.5, 0.5]];
        BTreeMap::from([
            ("w_in".to_string(), w_in.into_dyn()),
            ("b_in".to_string(), b_in.into_dyn()),
            ("w_out".to_string(), w_out.into_dyn()),
            ("b_out".to_string(), array![0.0, 1.0].into_dyn()),
            ("q".to_string(), array![1.0, 0.0, -1.0, 0.5].into_dyn()),
        ])
    }

    fn request(subsets: &str, vectors: &str) -> String {
        request_json(&format!(
            r#"{{"kind": "module_split", "block": {{"hidden_act": "gelu", "w_in": "w_in", "b_in": "b_in",
                "w_out": "w_out", "b_out": "b_out", "skip": null}}, "subsets": {subsets}, "laplacian_vectors": {vectors}}}"#
        ))
    }

    #[test]
    fn module_split_report_is_the_owner_result_field_for_field() {
        let tensors = tensors();
        let block = MlpBlockRequest {
            hidden_act: "gelu".to_string(),
            w_in: "w_in".to_string(),
            b_in: "b_in".to_string(),
            w_out: "w_out".to_string(),
            b_out: "b_out".to_string(),
            skip: None,
        };
        let form = normal_form(&tensors, &block).expect("owner normal form");
        assert_eq!(form.reads.nrows(), 4, "the negated pair merges into one unit");
        let blocks = form.additive_blocks(test_governor()).expect("owner blocks");
        let split = form.optimal_split(test_governor(), &blocks, &[0, 1]).expect("owner split");
        let product = blocks.laplacian_apply(vector(&tensors, "q").expect("q")).expect("owner laplacian");
        let expected = project(form, blocks, vec![split], vec![product]).expect("projection");
        let output = run_parameter_decomposition(&request("[[0, 1]]", r#"["q"]"#), &tensors, test_governor())
            .expect("surface run");
        assert_eq!(output, expected);
        let MpdResult::ModuleSplit(report) = &output.report.result else {
            panic!("expected a module split, got {:?}", output.report.result);
        };
        assert_eq!(report.normal_form.sources[3].len(), 2);
        assert!(report.blocks.finest.len() >= 2, "{:?}", report.blocks.finest);
    }

    #[test]
    fn a_module_split_the_owner_refuses_reaches_the_caller() {
        let tensors = tensors();
        assert!(run_parameter_decomposition(&request("[]", "[]"), &tensors, test_governor()).is_ok());
        for refused in [request("[[0, 0]]", "[]"), request("[[9]]", "[]")] {
            assert!(matches!(
                run_parameter_decomposition(&refused, &tensors, test_governor()),
                Err(MpdSurfaceError::ModuleSplit(_))
            ));
        }
        let mut short = tensors.clone();
        short.insert("q".to_string(), array![1.0].into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request("[]", r#"["q"]"#), &short, test_governor()),
            Err(MpdSurfaceError::ModuleSplit(_))
        ));
        assert!(matches!(
            run_parameter_decomposition(&request("[]", "[]").replace("\"gelu\"", "\"gelu_new\""), &tensors, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            run_parameter_decomposition(&request("[]", "[]").replace("\"skip\": null", "\"skip\": null, \"tau\": 0.1"), &tensors, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}

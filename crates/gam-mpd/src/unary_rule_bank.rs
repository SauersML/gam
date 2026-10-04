//! Complete two-node, one-input MatrixRule proposals. No fit-based inventory cuts.
//! Source evaluation is cached only as a construction optimization: serialized
//! artifacts contain the full body and bindings and ordinary decoders recompute it.
use crate::{
    artifact::{Artifact, Derived, OperatorLaw},
    attention_map::AttentionLayerMap,
    matrix_rule::{MatrixRule, Type, Value},
    matrix_rule_enumeration::{self as enumeration, Budget, Inventory, Source, Target},
    operator_program::{Operator, OperatorBody, exact_precision},
};
use ndarray::Array2;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug)]
pub struct TargetBinding {
    pub operator: usize,
    pub native_layer: usize,
    pub head: usize,
    pub reads: Vec<usize>,
    pub write: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_prefixes: u64,
    pub max_bodies: usize,
    pub max_candidates_including_native: u64,
    pub cache_bytes: u64,
    /// Bound on logical input, DAG-value, and fitting matrix buffers. SVD library
    /// scratch is additional; use a host allocation limit and record actual RSS.
    pub matrix_workspace_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub family: usize,
    pub source: usize,
}
#[derive(Clone, Debug)]
pub struct FitDiagnostic {
    pub amplitude: f32,
    pub native_weight_squared: f64,
    pub residual_weight_squared: f64,
    pub relative_weight_frobenius: f64,
    pub logical_workspace_bytes: u64,
}
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub unresolved: u64,
}

pub struct UnaryRuleBank<'a> {
    native: &'a Artifact,
    pub targets: Vec<TargetBinding>,
    pub inventory: Inventory,
    pub candidate_count: u64,
    limits: Limits,
    values: BTreeMap<(u64, Vec<u8>, usize), Result<Arc<Array2<f64>>, String>>,
    pub cache_stats: CacheStats,
}
fn bytes(rows: usize, cols: usize) -> Result<u64, String> {
    (rows as u64)
        .checked_mul(cols as u64)
        .and_then(|n| n.checked_mul(8))
        .ok_or("matrix byte count overflow".into())
}
fn type_bytes(ty: &Type) -> Result<u64, String> {
    match *ty {
        Type::Matrix { rows, cols } => bytes(rows, cols),
        Type::Vector { len } => bytes(len, 1),
    }
}
impl<'a> UnaryRuleBank<'a> {
    /// Every head of every declared native layer, with original graph lineage.
    pub fn all_attention(
        native: &'a Artifact,
        native_layers: &[usize],
        limits: Limits,
    ) -> Result<Self, String> {
        if native_layers
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != native_layers.len()
        {
            return Err("duplicate native layer ID".into());
        }
        let mut targets = vec![];
        for &layer in native_layers {
            let map = AttentionLayerMap::of(&native.program, layer)?;
            for head in &map.heads {
                targets.push(TargetBinding {
                    operator: head.output_operator,
                    native_layer: layer,
                    head: head.head,
                    reads: map.native_reads().to_vec(),
                    write: map.output,
                });
            }
        }
        Self::new(native, targets, limits)
    }
    /// Generic target-binding entry point for small independently executable tests.
    pub fn new(
        native: &'a Artifact,
        targets: Vec<TargetBinding>,
        limits: Limits,
    ) -> Result<Self, String> {
        if !native.derived.is_empty()
            || !native.exceptions.is_empty()
            || !native.blocks.is_empty()
            || !native.has_f32_literals()
        {
            return Err("unary inventory requires unmodified native f32 literal artifact".into());
        }
        if targets.is_empty() {
            return Err("empty declared target inventory".into());
        }
        let mut sources = vec![];
        for (id, operator) in native.program.operators.iter().enumerate() {
            let ty = match &operator.body {
                OperatorBody::Identity => continue, // primitive identity has no learned payload.
                OperatorBody::Diagonal { values, .. } => Type::Vector { len: values.len() },
                OperatorBody::Dense { .. } | OperatorBody::LowRank { .. } => Type::Matrix {
                    rows: operator.rows.width(),
                    cols: operator.cols.width(),
                },
            };
            // All native learned operators are independently transmitted sources.
            sources.push(Source {
                id,
                ty,
                dependencies: vec![],
            });
        }
        let ts: Vec<_> = targets
            .iter()
            .map(|t| {
                native
                    .program
                    .operators
                    .get(t.operator)
                    .map(|op| Target {
                        id: t.operator,
                        ty: Type::Matrix {
                            rows: op.rows.width(),
                            cols: op.cols.width(),
                        },
                    })
                    .ok_or("target operator absent".to_string())
            })
            .collect::<Result<_, _>>()?;
        let budget = Budget {
            max_nodes: 2,
            max_c32: u64::MAX,
            max_prefixes: limits.max_prefixes,
            max_bodies: limits.max_bodies,
            scale_coefficients: vec![],
        };
        let inventory = enumeration::inventory(
            &ts,
            &sources,
            1,
            &budget,
            limits.max_candidates_including_native.saturating_sub(1),
        )?;
        if !inventory.complete {
            return Err(format!(
                "metadata inventory incomplete: {:?}; known binding lower bound {}",
                inventory.stop_reason, inventory.binding_count
            ));
        }
        let candidate_count = inventory
            .binding_count
            .checked_add(1)
            .ok_or("candidate cardinality overflow")?;
        if candidate_count > limits.max_candidates_including_native {
            return Err("complete unary bank exceeds declared cardinality budget".into());
        }
        Ok(Self {
            native,
            targets,
            inventory,
            candidate_count,
            limits,
            values: BTreeMap::new(),
            cache_stats: CacheStats::default(),
        })
    }
    /// Explicit controls for every declared target, never a selected subset.
    pub fn zero_controls(&self) -> std::ops::Range<usize> {
        0..self.targets.len()
    }
    pub fn cardinality_with_zero_controls(&self) -> Result<u64, String> {
        self.candidate_count
            .checked_add(self.targets.len() as u64)
            .ok_or("control cardinality overflow".into())
    }
    /// Discard exactly one native head with a sparse empty operator: zero numeric
    /// payload, original interfaces/nodes/intervention places and same binding.
    pub fn zero_candidate(&self, target_index: usize) -> Result<Artifact, String> {
        let target = self
            .targets
            .get(target_index)
            .ok_or("zero-control target outside complete inventory")?;
        let original = &self.native.program.operators[target.operator];
        let values = Array2::zeros((original.rows.width(), original.cols.width()));
        let present = Array2::from_elem(
            (original.rows.group_count(), original.cols.group_count()),
            false,
        );
        let operator = Operator::blocks(
            original.name.clone(),
            original.rows.clone(),
            original.cols.clone(),
            values,
            present,
            exact_precision([0.0]).map_err(|e| e.to_string())?,
            original.provenance.clone(),
        )
        .map_err(|e| e.to_string())?;
        let mut artifact = self.native.clone();
        artifact.program.operators[target.operator] = Arc::new(operator);
        artifact.bind(
            &format!(
                "generic unary attention {} post-residual boundary",
                target.native_layer
            ),
            &target.reads,
            target.write,
        )
    }
    pub fn choices(&self) -> impl Iterator<Item = Choice> + '_ {
        self.inventory
            .families
            .iter()
            .enumerate()
            .flat_map(|(family, f)| {
                f.source_pools[0]
                    .iter()
                    .copied()
                    .map(move |source| Choice { family, source })
            })
    }
    pub fn target(&self, choice: Choice) -> Result<&TargetBinding, String> {
        let family = self
            .inventory
            .families
            .get(choice.family)
            .ok_or("family outside declared inventory")?;
        self.targets
            .iter()
            .find(|t| t.operator == family.target)
            .ok_or("target binding absent".into())
    }
    pub fn body(&self, choice: Choice) -> Result<&MatrixRule, String> {
        let family = self
            .inventory
            .families
            .get(choice.family)
            .ok_or("family outside declared inventory")?;
        if !family.source_pools[0].contains(&choice.source) {
            return Err("source binding outside declared inventory".into());
        }
        Ok(&family.body.rule)
    }
    fn source_value(&mut self, choice: Choice) -> Result<(Arc<Array2<f64>>, u64), String> {
        let body = self.body(choice)?.clone();
        let encoded = body.encode()?;
        let key = (
            encoded.len_bits(),
            encoded.packed_bytes().to_vec(),
            choice.source,
        );
        let target = &self.native.program.operators[self.target(choice)?.operator];
        let mut workspace = type_bytes(&body.inputs[0])?;
        for ty in body.types()? {
            workspace = workspace
                .checked_add(type_bytes(&ty)?)
                .ok_or("workspace byte count overflow")?;
        }
        // Scaled prediction, native matrix, and temporary residual metric buffer.
        workspace = workspace
            .checked_add(
                bytes(target.rows.width(), target.cols.width())?
                    .checked_mul(3)
                    .ok_or("workspace overflow")?,
            )
            .ok_or("workspace overflow")?;
        if workspace > self.limits.matrix_workspace_bytes {
            self.cache_stats.unresolved += 1;
            return Err(format!(
                "logical matrix workspace {workspace} exceeds declared {}; unresolved",
                self.limits.matrix_workspace_bytes
            ));
        }
        if let Some(value) = self.values.get(&key) {
            self.cache_stats.hits += 1;
            return value.clone().map(|x| (x, workspace));
        }
        self.cache_stats.misses += 1;
        let output_bytes = type_bytes(&body.types()?[body.output])?;
        let value = if output_bytes
            > self
                .limits
                .cache_bytes
                .saturating_sub(self.cache_stats.bytes)
        {
            Err(format!(
                "source cache budget {} exhausted at {} bytes; unresolved",
                self.limits.cache_bytes, self.cache_stats.bytes
            ))
        } else {
            let source = &self.native.program.operators[choice.source];
            let input = match body.inputs[0] {
                Type::Matrix { .. } => Value::Matrix(source.matrix()),
                Type::Vector { .. } => Value::Vector(
                    source
                        .diagonal()
                        .ok_or("vector source lacks native diagonal")?,
                ),
            };
            match body.evaluate(&[input]) {
                Ok(Value::Matrix(x)) => Ok(Arc::new(x)),
                Ok(Value::Vector(_)) => Err("matrix target received vector expression".into()),
                Err(e) => Err(format!("undefined expression: {e}; unresolved")),
            }
        };
        match &value {
            Ok(x) => self.cache_stats.bytes += bytes(x.nrows(), x.ncols())?,
            Err(_) => self.cache_stats.unresolved += 1,
        }
        self.values.insert(key, value.clone());
        value.map(|x| (x, workspace))
    }
    /// Price-only skeleton. Its target values are intentionally not evaluated.
    /// C32 omits derived target payloads and prices every amplitude at32bits, so
    /// this price remains exact even when fitting/evaluation is unresolved.
    pub fn priced_skeleton(&self, choice: Choice) -> Result<Artifact, String> {
        let target = self.target(choice)?;
        let body = Arc::new(self.body(choice)?.clone());
        let mut artifact = self.native.clone();
        artifact.derived.push(Derived {
            operator: target.operator,
            law: OperatorLaw::Expression {
                body,
                sources: vec![choice.source],
            },
            scale: 0.0,
            residual: vec![],
        });
        artifact.bind(
            &format!(
                "generic unary attention {} post-residual boundary",
                target.native_layer
            ),
            &target.reads,
            target.write,
        )
    }
    /// Fit the one independently priced f32 amplitude to native matrix entries.
    /// Rank-zero residual only: no rows/factors are omitted from the price.
    pub fn candidate(&mut self, choice: Choice) -> Result<(Artifact, FitDiagnostic), String> {
        let target = self.target(choice)?.clone();
        let body = Arc::new(self.body(choice)?.clone());
        let (prediction, workspace) = self.source_value(choice)?;
        let original = &self.native.program.operators[target.operator];
        let native = original.matrix();
        let denominator: f64 = prediction.iter().map(|v| v * v).sum();
        let numerator: f64 = native
            .iter()
            .zip(prediction.iter())
            .map(|(w, p)| w * p)
            .sum();
        if !denominator.is_finite() || !numerator.is_finite() {
            return Err("amplitude fit overflow; unresolved".into());
        }
        let amplitude = if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator) as f32
        };
        if !amplitude.is_finite() {
            return Err("amplitude not finite f32; unresolved".into());
        }
        let values = prediction.mapv(|p| p * f64::from(amplitude));
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        // Same arithmetic and operator construction as artifact::compute_derived;
        // independent ordinary decoding remains required for later acceptance.
        let op = Operator::dense(
            original.name.clone(),
            original.rows.clone(),
            original.cols.clone(),
            values,
            precision,
            original.provenance.clone(),
        )
        .map_err(|e| e.to_string())?;
        let reconstructed = op.matrix();
        let native_weight_squared: f64 = native.iter().map(|v| v * v).sum();
        let residual_weight_squared: f64 = native
            .iter()
            .zip(reconstructed.iter())
            .map(|(w, p)| (w - p) * (w - p))
            .sum();
        if !native_weight_squared.is_finite() || !residual_weight_squared.is_finite() {
            return Err("weight metric overflow; unresolved".into());
        }
        let relative_weight_frobenius = if native_weight_squared == 0.0 {
            if residual_weight_squared == 0.0 {
                0.0
            } else {
                return Err("relative weight metric has zero denominator; unresolved".into());
            }
        } else {
            (residual_weight_squared / native_weight_squared).sqrt()
        };
        let mut artifact = self.native.clone();
        artifact.program.operators[target.operator] = Arc::new(op);
        artifact.derived.push(Derived {
            operator: target.operator,
            law: OperatorLaw::Expression {
                body,
                sources: vec![choice.source],
            },
            scale: amplitude,
            residual: vec![],
        });
        let artifact = artifact.bind(
            &format!(
                "generic unary attention {} post-residual boundary",
                target.native_layer
            ),
            &target.reads,
            target.write,
        )?;
        Ok((
            artifact,
            FitDiagnostic {
                amplitude,
                native_weight_squared,
                residual_weight_squared,
                relative_weight_frobenius,
                logical_workspace_bytes: workspace,
            },
        ))
    }
}
#[cfg(test)]
#[path = "unary_rule_bank_tests.rs"]
mod tests;

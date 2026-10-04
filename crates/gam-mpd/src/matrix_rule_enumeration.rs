//! Metadata-only exhaustive enumeration of a bounded, ordered matrix DAG grammar.
//! Completion certifies this declared grammar only, never fidelity or optimality.
use crate::matrix_rule::{MatrixRule, Node, PinvConvention, Type};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct Budget {
    pub max_nodes: usize,
    pub max_c32: u64,
    pub max_prefixes: u64,
    pub max_bodies: usize,
    /// Internal arithmetic constants are an explicit finite inventory, not fitted here.
    pub scale_coefficients: Vec<f32>,
}
#[derive(Clone, Debug)]
pub struct Body {
    pub rule: MatrixRule,
    /// Canonical input slot -> caller's original input slot. Apply to bindings too.
    pub input_permutation: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct Enumeration {
    pub bodies: Vec<Body>,
    pub visited_prefixes: u64,
    pub complete: bool,
    pub stop_reason: Option<&'static str>,
}
fn refs(n: &Node) -> Vec<usize> {
    match *n {
        Node::Param { .. } => vec![],
        Node::MatMul { left, right } => vec![left, right],
        Node::Divide {
            numerator,
            denominator,
        } => vec![numerator, denominator],
        Node::Transpose { input }
        | Node::Diag { input }
        | Node::Reciprocal { input }
        | Node::Pinv { input, .. }
        | Node::Scale { input, .. } => vec![input],
    }
}
fn remap(n: &Node, m: &[usize]) -> Node {
    match *n {
        Node::Param { index } => Node::Param { index },
        Node::Transpose { input } => Node::Transpose { input: m[input] },
        Node::Diag { input } => Node::Diag { input: m[input] },
        Node::Reciprocal { input } => Node::Reciprocal { input: m[input] },
        Node::Pinv { input, convention } => Node::Pinv {
            input: m[input],
            convention,
        },
        Node::Scale { input, coefficient } => Node::Scale {
            input: m[input],
            coefficient,
        },
        Node::MatMul { left, right } => Node::MatMul {
            left: m[left],
            right: m[right],
        },
        Node::Divide {
            numerator,
            denominator,
        } => Node::Divide {
            numerator: m[numerator],
            denominator: m[denominator],
        },
    }
}
/// Only topological renumbering and parameter-slot permutation; no algebraic rewrites.
pub fn canonicalize(rule: &MatrixRule) -> Result<Body, String> {
    rule.types()?;
    fn visit(i: usize, r: &MatrixRule, seen: &mut BTreeSet<usize>, order: &mut Vec<usize>) {
        if !seen.insert(i) {
            return;
        }
        for j in refs(&r.nodes[i]) {
            visit(j, r, seen, order);
        }
        order.push(i);
    }
    let mut order = vec![];
    visit(rule.output, rule, &mut BTreeSet::new(), &mut order);
    let mut inputs = vec![];
    let mut permutation = vec![];
    let mut slots = BTreeMap::new();
    let mut map = vec![usize::MAX; rule.nodes.len()];
    let mut nodes = vec![];
    for old in order {
        let mut node = remap(&rule.nodes[old], &map);
        if let Node::Param { index } = node {
            let slot = *slots.entry(index).or_insert_with(|| {
                let s = inputs.len();
                inputs.push(rule.inputs[index].clone());
                permutation.push(index);
                s
            });
            node = Node::Param { index: slot };
        }
        map[old] = nodes.len();
        nodes.push(node);
    }
    Ok(Body {
        rule: MatrixRule {
            inputs,
            nodes,
            output: map[rule.output],
        },
        input_permutation: permutation,
    })
}
/// Fixed input signature; every declared input and node must participate in output.
/// Parameter reads precede arithmetic. Independent parameter reads can always be
/// moved to the front without changing arithmetic order or source bindings.
pub fn enumerate(inputs: &[Type], output: &Type, budget: &Budget) -> Result<Enumeration, String> {
    if inputs.is_empty() || budget.max_nodes < inputs.len() {
        return Err("node budget must include all nonempty inputs".into());
    }
    if budget.scale_coefficients.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite declared coefficient".into());
    }
    let mut result = Enumeration {
        bodies: vec![],
        visited_prefixes: 0,
        complete: true,
        stop_reason: None,
    };
    let mut keys = BTreeSet::new();
    let mut rule = MatrixRule {
        inputs: inputs.to_vec(),
        nodes: (0..inputs.len())
            .map(|index| Node::Param { index })
            .collect(),
        output: inputs.len() - 1,
    };
    rule.types()?;
    fn walk(
        r: &mut MatrixRule,
        out: &Type,
        b: &Budget,
        res: &mut Enumeration,
        keys: &mut BTreeSet<(u64, Vec<u8>)>,
    ) -> Result<(), String> {
        if !res.complete {
            return Ok(());
        }
        if res.visited_prefixes >= b.max_prefixes {
            res.complete = false;
            res.stop_reason = Some("max_prefixes; unexplored grammar cardinality unknown");
            return Ok(());
        }
        res.visited_prefixes += 1;
        r.output = r.nodes.len() - 1;
        let types = r.types()?;
        let canonical = canonicalize(r)?;
        if canonical.rule.nodes.len() == r.nodes.len()
            && canonical.rule.inputs.len() == r.inputs.len()
            && types[r.output] == *out
            && canonical.rule.cost()?.c32() <= b.max_c32
        {
            let encoded = canonical.rule.encode()?;
            let key = (encoded.len_bits(), encoded.packed_bytes().to_vec());
            if !keys.contains(&key) {
                if res.bodies.len() >= b.max_bodies {
                    res.complete = false;
                    res.stop_reason = Some("max_bodies; unexplored grammar cardinality unknown");
                    return Ok(());
                }
                keys.insert(key);
                res.bodies.push(canonical);
            }
        }
        if r.nodes.len() == b.max_nodes {
            return Ok(());
        }
        // Each future binary node can merge at most two currently unused roots.
        let mut used = vec![false; r.nodes.len()];
        for node in &r.nodes {
            for i in refs(node) {
                used[i] = true;
            }
        }
        if used.iter().filter(|u| !**u).count() > b.max_nodes - r.nodes.len() + 1 {
            return Ok(());
        }
        let n = r.nodes.len();
        let mut options = vec![];
        for i in 0..n {
            options.push(Node::Reciprocal { input: i });
            for &coefficient in &b.scale_coefficients {
                options.push(Node::Scale {
                    input: i,
                    coefficient,
                });
            }
            match types[i] {
                Type::Matrix { .. } => {
                    options.push(Node::Transpose { input: i });
                    options.push(Node::Pinv {
                        input: i,
                        convention: PinvConvention::SvdResolutionBand,
                    });
                }
                Type::Vector { .. } => options.push(Node::Diag { input: i }),
            }
            for j in 0..n {
                if types[i] == types[j] {
                    options.push(Node::Divide {
                        numerator: i,
                        denominator: j,
                    });
                }
                if let (Type::Matrix { cols, .. }, Type::Matrix { rows, .. }) =
                    (&types[i], &types[j])
                {
                    if cols == rows {
                        options.push(Node::MatMul { left: i, right: j });
                    }
                }
            }
        }
        for node in options {
            r.nodes.push(node);
            walk(r, out, b, res, keys)?;
            r.nodes.pop();
            if !res.complete {
                break;
            }
        }
        Ok(())
    }
    walk(&mut rule, output, budget, &mut result, &mut keys)?;
    Ok(result)
}
#[derive(Clone, Debug)]
pub struct Source {
    pub id: usize,
    pub ty: Type,
    pub dependencies: Vec<usize>,
}
/// Source eligibility excludes the target and every transitive dependent. Missing
/// metadata or dependency cycles are errors, never silently omitted candidates.
pub fn eligible_sources(sources: &[Source], target: usize) -> Result<Vec<&Source>, String> {
    let map: BTreeMap<_, _> = sources.iter().map(|s| (s.id, s)).collect();
    if map.len() != sources.len() {
        return Err("duplicate source ID".into());
    }
    fn depends(
        id: usize,
        target: usize,
        map: &BTreeMap<usize, &Source>,
        active: &mut BTreeSet<usize>,
    ) -> Result<bool, String> {
        let s = map.get(&id).ok_or("missing dependency metadata")?;
        if !active.insert(id) {
            return Err("dependency cycle".into());
        }
        let mut result = id == target;
        for &d in &s.dependencies {
            result |= depends(d, target, map, active)?;
        }
        active.remove(&id);
        Ok(result)
    }
    sources
        .iter()
        .filter_map(
            |s| match depends(s.id, target, &map, &mut BTreeSet::new()) {
                Ok(false) => Some(Ok(s)),
                Ok(true) => None,
                Err(e) => Some(Err(e)),
            },
        )
        .collect()
}
/// Lazy mixed-radix binding inventory. Repeated source IDs across slots are legal.
pub struct Bindings {
    pools: Vec<Vec<usize>>,
    next: u64,
    pub count: u64,
}
impl Bindings {
    pub fn new(body: &MatrixRule, sources: &[&Source], max_bindings: u64) -> Result<Self, String> {
        let pools: Vec<Vec<usize>> = body
            .inputs
            .iter()
            .map(|ty| {
                sources
                    .iter()
                    .filter(|s| s.ty == *ty)
                    .map(|s| s.id)
                    .collect()
            })
            .collect();
        let count = pools
            .iter()
            .try_fold(1u64, |a, p| a.checked_mul(p.len() as u64))
            .ok_or("binding cardinality overflow; unresolved")?;
        if count > max_bindings {
            return Err(format!(
                "{count} bindings exceed declared budget {max_bindings}; unresolved"
            ));
        }
        Ok(Self {
            pools,
            next: 0,
            count,
        })
    }
}
impl Iterator for Bindings {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let mut i = self.next;
        self.next += 1;
        Some(
            self.pools
                .iter()
                .map(|p| {
                    let value = p[(i % p.len() as u64) as usize];
                    i /= p.len() as u64;
                    value
                })
                .collect(),
        )
    }
}
#[cfg(test)]
#[path = "matrix_rule_enumeration_tests.rs"]
mod tests;

#[derive(Clone, Debug)]
pub struct Target {
    pub id: usize,
    pub ty: Type,
}
#[derive(Clone, Debug)]
pub struct Family {
    pub target: usize,
    pub body: Body,
    /// Canonical input slots, including repeated choices; no materialized tuples.
    pub source_pools: Vec<Vec<usize>>,
    pub binding_count: u64,
}
#[derive(Clone, Debug)]
pub struct Inventory {
    pub families: Vec<Family>,
    /// Lower bound on the full inventory if incomplete; exact count if complete.
    pub binding_count: u64,
    pub visited_prefixes: u64,
    pub complete: bool,
    pub stop_reason: Option<String>,
    /// Exact externally declared scope, independent of numerical measurements.
    pub targets: Vec<Target>,
    pub sources: Vec<Source>,
    pub grammar_budget: Budget,
    pub max_inputs: usize,
}
fn type_key(t: &Type) -> (u8, usize, usize) {
    match *t {
        Type::Matrix { rows, cols } => (0, rows, cols),
        Type::Vector { len } => (1, len, 0),
    }
}
/// All declared targets, all eligible source type signatures up to max_inputs,
/// and every connected body under Budget. This API does not select native heads;
/// callers must record the entire native target/source inventory they declare.
/// Type signatures are multisets: source-slot permutations are already represented
/// by ordered DAG edges and canonical binding permutations, not algebraic rewrites.
pub fn inventory(
    targets: &[Target],
    sources: &[Source],
    max_inputs: usize,
    budget: &Budget,
    max_bindings: u64,
) -> Result<Inventory, String> {
    if max_inputs == 0 {
        return Err("max_inputs must be positive".into());
    }
    if targets.iter().map(|t| t.id).collect::<BTreeSet<_>>().len() != targets.len() {
        return Err("duplicate target ID".into());
    }
    let mut result = Inventory {
        families: vec![],
        binding_count: 0,
        visited_prefixes: 0,
        complete: true,
        stop_reason: None,
        targets: targets.to_vec(),
        sources: sources.to_vec(),
        grammar_budget: budget.clone(),
        max_inputs,
    };
    for target in targets {
        let eligible = eligible_sources(sources, target.id)?;
        let types: BTreeMap<_, _> = eligible
            .iter()
            .map(|s| (type_key(&s.ty), s.ty.clone()))
            .collect();
        let types: Vec<_> = types.into_values().collect();
        let mut signatures = vec![];
        fn signatures_of(
            types: &[Type],
            left: usize,
            start: usize,
            prefix: &mut Vec<Type>,
            out: &mut Vec<Vec<Type>>,
        ) {
            if left == 0 {
                out.push(prefix.clone());
                return;
            }
            for i in start..types.len() {
                prefix.push(types[i].clone());
                signatures_of(types, left - 1, i, prefix, out);
                prefix.pop();
            }
        }
        // Signature metadata itself is guarded before allocating a large product.
        let mut signature_count = 1u64;
        let mut signature_total = 0u64;
        for p in 1..=max_inputs.min(budget.max_nodes) {
            if types.is_empty() {
                break;
            }
            signature_count = signature_count
                .checked_mul((types.len() + p - 1) as u64)
                .and_then(|n| n.checked_div(p as u64))
                .ok_or("signature cardinality overflow; unresolved")?;
            signature_total = signature_total
                .checked_add(signature_count)
                .ok_or("signature cardinality overflow; unresolved")?;
            if signature_total > budget.max_prefixes.saturating_sub(result.visited_prefixes) {
                result.complete = false;
                result.stop_reason =
                    Some("type-signature inventory exceeds prefix budget; unresolved".into());
                return Ok(result);
            }
            signatures_of(&types, p, 0, &mut vec![], &mut signatures);
        }
        let mut keys = BTreeSet::new();
        for signature in signatures {
            let mut remaining = budget.clone();
            remaining.max_prefixes = budget.max_prefixes.saturating_sub(result.visited_prefixes);
            remaining.max_bodies = budget.max_bodies.saturating_sub(result.families.len());
            let enumeration = enumerate(&signature, &target.ty, &remaining)?;
            result.visited_prefixes += enumeration.visited_prefixes;
            for body in enumeration.bodies {
                let encoded = body.rule.encode()?;
                if !keys.insert((encoded.len_bits(), encoded.packed_bytes().to_vec())) {
                    continue;
                }
                let pools: Vec<Vec<usize>> = body
                    .rule
                    .inputs
                    .iter()
                    .map(|ty| {
                        eligible
                            .iter()
                            .filter(|s| s.ty == *ty)
                            .map(|s| s.id)
                            .collect()
                    })
                    .collect();
                let Some(count) = pools
                    .iter()
                    .try_fold(1u64, |a, p| a.checked_mul(p.len() as u64))
                else {
                    result.complete = false;
                    result.stop_reason = Some("binding cardinality overflow; unresolved".into());
                    return Ok(result);
                };
                let Some(total) = result.binding_count.checked_add(count) else {
                    result.complete = false;
                    result.stop_reason =
                        Some("total binding cardinality overflow; unresolved".into());
                    return Ok(result);
                };
                if total > max_bindings {
                    result.complete = false;
                    result.stop_reason = Some(format!(
                        "known next family adds {count} bindings exceeding declared {max_bindings}; unresolved"
                    ));
                    return Ok(result);
                }
                result.binding_count = total;
                result.families.push(Family {
                    target: target.id,
                    body,
                    source_pools: pools,
                    binding_count: count,
                });
            }
            if !enumeration.complete {
                result.complete = false;
                result.stop_reason = enumeration.stop_reason.map(str::to_owned);
                return Ok(result);
            }
        }
    }
    Ok(result)
}

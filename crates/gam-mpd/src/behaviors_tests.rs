#![cfg(test)]
//! The partition code and its search on a table fitter whose answer is written down: a model that
//! reads `x` under task 0 and answers a constant under task 1 is one table over `(task, x)` to a
//! single program and two short programs to the partition, recognized by `task`.

use super::behaviors::{
    BehaviorError, GroupFitter, ProgramParts, Recognizer, RowFeature, Within, atom_seeds, discover,
    fit_recognizer, joint_program_bits,
};
use super::codec::{fixed_index_len_bits, prefix_integer_len_bits, subset_code_len_bits};

const CLASSES: usize = 8;
const XS: usize = 8;
/// Enough rows that the task-0 table pays for itself against a constant.
const REPEATS: usize = 60;
/// The code of one table entry: a declared lattice of 2⁻⁸ over a fixed range.
const ENTRY_BITS: u64 = 12;

/// Which inputs a table reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reads {
    Nothing,
    X,
    TaskAndX,
}

#[derive(Clone, Debug, PartialEq)]
struct Table {
    reads: Reads,
    /// One distribution per cell.
    cells: Vec<Vec<f64>>,
}

struct TableFitter {
    task: Vec<u32>,
    x: Vec<u32>,
    model: Vec<Vec<f64>>,
}

fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b).ln()).sum::<f64>() / std::f64::consts::LN_2
}

impl TableFitter {
    fn planted() -> Self {
        let (mut task, mut x, mut model) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..REPEATS {
            for t in 0..2u32 {
                for v in 0..XS as u32 {
                    let peak = if t == 0 { v as usize } else { 3 };
                    let mut p = vec![0.02; CLASSES];
                    p[peak] = 1.0 - 0.02 * (CLASSES - 1) as f64;
                    task.push(t);
                    x.push(v);
                    model.push(p);
                }
            }
        }
        Self { task, x, model }
    }

    fn cell(&self, reads: Reads, row: usize) -> usize {
        match reads {
            Reads::Nothing => 0,
            Reads::X => self.x[row] as usize,
            Reads::TaskAndX => self.task[row] as usize * XS + self.x[row] as usize,
        }
    }

    fn table_on(&self, reads: Reads, rows: &[usize]) -> Table {
        let cells = match reads {
            Reads::Nothing => 1,
            Reads::X => XS,
            Reads::TaskAndX => 2 * XS,
        };
        let mut sums = vec![vec![0.0; CLASSES]; cells];
        let mut counts = vec![0usize; cells];
        for &row in rows {
            let c = self.cell(reads, row);
            counts[c] += 1;
            for (s, p) in sums[c].iter_mut().zip(&self.model[row]) {
                *s += p;
            }
        }
        for (sum, n) in sums.iter_mut().zip(&counts) {
            for s in sum.iter_mut() {
                *s = if *n == 0 { 1.0 / CLASSES as f64 } else { *s / *n as f64 };
            }
        }
        Table { reads, cells: sums }
    }

    fn table_bits(table: &Table) -> u64 {
        2 + table.cells.len() as u64 * CLASSES as u64 * ENTRY_BITS
    }
}

impl GroupFitter for TableFitter {
    type Program = Table;
    type Certificate = f64;

    fn rows(&self) -> usize {
        self.model.len()
    }

    fn fit(&mut self, rows: &[usize], within: Option<Within<'_, Table>>) -> Result<(Table, f64), BehaviorError> {
        // The enclosing group's table is tried first, so it wins ties.
        let mut order = vec![Reads::Nothing, Reads::X, Reads::TaskAndX];
        if let Some(parent) = within {
            order.retain(|r| *r != parent.program.reads);
            order.insert(0, parent.program.reads);
        }
        let mut best: Option<(f64, f64, Table)> = None;
        for reads in order {
            let table = self.table_on(reads, rows);
            let data: f64 = rows.iter().map(|&r| kl(&self.model[r], &table.cells[self.cell(reads, r)])).sum();
            let total = Self::table_bits(&table) as f64 + data;
            if best.as_ref().is_none_or(|(b, _, _)| total < *b) {
                best = Some((total, data, table));
            }
        }
        best.map(|(_, data, table)| (table, data)).ok_or_else(|| BehaviorError::Fit("no table".to_string()))
    }

    fn row_bits(&mut self, program: &Table) -> Result<Vec<f64>, BehaviorError> {
        Ok((0..self.rows()).map(|r| kl(&self.model[r], &program.cells[self.cell(program.reads, r)])).collect())
    }

    fn parts(&self, program: &Table) -> Result<ProgramParts, BehaviorError> {
        let bits = Self::table_bits(program);
        Ok(ProgramParts { bits, operators: vec![(format!("{program:?}"), bits - 2)] })
    }

    fn certify(&mut self, program: &Table, rows: &[usize]) -> Result<(f64, f64), BehaviorError> {
        let bits = self.row_bits(program)?;
        let data: f64 = rows.iter().map(|r| bits[*r]).sum();
        let worst = rows.iter().map(|r| bits[*r]).fold(0.0, f64::max);
        Ok((worst, data))
    }
}

fn features(fitter: &TableFitter) -> Vec<RowFeature> {
    vec![
        RowFeature { name: "task".to_string(), alphabet: 2, values: fitter.task.clone(), definition_bits: None },
        RowFeature { name: "x".to_string(), alphabet: XS, values: fitter.x.clone(), definition_bits: None },
    ]
}

#[test]
fn a_one_leaf_recognizer_is_the_enumerative_code_of_the_labels() {
    let labels = vec![0, 1, 1, 0, 2, 1, 0, 0];
    let (tree, bits) = fit_recognizer(&labels, 3, &[]).expect("fits");
    assert_eq!(tree, Recognizer::Leaf { default: 0 });
    // One node bit, the default among three, then group 1's three rows among eight and group 2's
    // one row among the five left.
    let expected = 1
        + u64::from(fixed_index_len_bits(3).expect("width"))
        + subset_code_len_bits(8, 3).expect("code")
        + subset_code_len_bits(5, 1).expect("code");
    assert_eq!(bits, expected);
}

#[test]
fn a_rule_that_recognizes_the_groups_leaves_nothing_to_enumerate() {
    let fitter = TableFitter::planted();
    let labels: Vec<usize> = fitter.task.iter().map(|t| *t as usize).collect();
    let (tree, bits) = fit_recognizer(&labels, 2, &features(&fitter)).expect("fits");
    let Recognizer::Split { feature: 0, equal, other, .. } = &tree else { panic!("the rule tests task, got {tree:?}") };
    assert!(matches!(**equal, Recognizer::Leaf { .. }) && matches!(**other, Recognizer::Leaf { .. }));
    // The split, its two pure leaves (each its default and an empty subset of the other group).
    let leaf = u64::from(fixed_index_len_bits(2).expect("width")) + subset_code_len_bits(fitter.task.len() / 2, 0).expect("code");
    let split = u64::from(fixed_index_len_bits(2).expect("width")) * 2;
    assert_eq!(bits, 1 + split + 2 * (1 + leaf));
}

#[test]
fn a_computed_predicate_is_read_only_when_it_pays_for_its_definition() {
    let fitter = TableFitter::planted();
    let labels: Vec<usize> = fitter.task.iter().map(|t| *t as usize).collect();
    for (definition, read) in [(4u64, true), (100_000, false)] {
        let predicate = RowFeature {
            name: "computed task".to_string(),
            alphabet: 2,
            values: fitter.task.clone(),
            definition_bits: Some(definition),
        };
        let (tree, _) = fit_recognizer(&labels, 2, &[predicate]).expect("fits");
        assert_eq!(matches!(tree, Recognizer::Split { .. }), read, "definition {definition}");
    }
}

#[test]
fn a_shared_operator_is_sent_once() {
    let a = ProgramParts { bits: 100, operators: vec![("E".to_string(), 60), ("U".to_string(), 30)] };
    let b = ProgramParts { bits: 80, operators: vec![("E".to_string(), 60), ("V".to_string(), 10)] };
    let (library, conditional) = joint_program_bits(&[&a, &b]).expect("codes");
    assert_eq!(library, prefix_integer_len_bits(2).expect("code") + 60);
    // Each program: one flag per operator slot, E replaced by an index into a one-entry library.
    assert_eq!(conditional, vec![100 + 2 - 60, 80 + 2 - 60]);
    let (alone, unshared) = joint_program_bits(&[&a]).expect("codes");
    assert_eq!((alone, unshared), (prefix_integer_len_bits(1).expect("code"), vec![100]));
}

#[test]
fn the_planted_task_split_is_discovered_and_recognized_by_task() {
    let mut fitter = TableFitter::planted();
    let features = features(&fitter);
    let seeds = atom_seeds(&features);
    let discovery = discover(&mut fitter, &features, &seeds).expect("discovers");
    assert_eq!(discovery.behaviours.len(), 2, "moves {:?}", discovery.moves);
    assert!(discovery.code.total() < discovery.one_group.total());
    for behaviour in &discovery.behaviours {
        let task = fitter.task[behaviour.rows[0]];
        assert!(behaviour.rows.iter().all(|r| fitter.task[*r] == task));
        assert_eq!(behaviour.rows.len(), fitter.task.len() / 2);
        let expected = if task == 0 { Reads::X } else { Reads::Nothing };
        assert_eq!(behaviour.program.reads, expected);
        assert_eq!(behaviour.rule.len(), 1);
        assert_eq!(behaviour.rule[0].tests.len(), 1);
        assert_eq!(behaviour.rule[0].tests[0].feature, "task");
        assert_eq!(behaviour.rule[0].own_rows, behaviour.rule[0].leaf_rows);
    }
    // The task-0 table is the longer program: the general part.
    let general: Vec<u32> =
        discovery.behaviours.iter().filter(|b| b.general).map(|b| fitter.task[b.rows[0]]).collect();
    assert_eq!(general, vec![0]);
}

#[test]
fn a_model_with_no_region_structure_stays_one_behaviour() {
    let mut fitter = TableFitter::planted();
    // Both tasks read x the same way: nothing is shorter on either task's rows.
    for row in 0..fitter.model.len() {
        let mut p = vec![0.02; CLASSES];
        p[fitter.x[row] as usize] = 1.0 - 0.02 * (CLASSES - 1) as f64;
        fitter.model[row] = p;
    }
    let features = features(&fitter);
    let seeds = atom_seeds(&features);
    let discovery = discover(&mut fitter, &features, &seeds).expect("discovers");
    assert_eq!(discovery.behaviours.len(), 1, "moves {:?}", discovery.moves);
    assert_eq!(discovery.behaviours[0].rule[0].render(), "always");
}

#[test]
fn two_disjoint_laws_are_one_behaviour() {
    let mut fitter = TableFitter::planted();
    // Each task reads x by its own law. The split's two exact tables over x cost as much as the
    // one table over (task, x), and the one group may also pay data bits instead (a table over x
    // coding the tasks' mixture), so splitting pays nothing: two laws are not two behaviours.
    for row in 0..fitter.model.len() {
        let shift = if fitter.task[row] == 0 { 0 } else { 3 };
        let mut p = vec![0.02; CLASSES];
        p[(fitter.x[row] as usize + shift) % CLASSES] = 1.0 - 0.02 * (CLASSES - 1) as f64;
        fitter.model[row] = p;
    }
    let features = features(&fitter);
    let seeds = atom_seeds(&features);
    let discovery = discover(&mut fitter, &features, &seeds).expect("discovers");
    assert_eq!(discovery.behaviours.len(), 1, "moves {:?}", discovery.moves);
    let split_programs = 2 * (2 + XS as u64 * CLASSES as u64 * ENTRY_BITS);
    assert!(discovery.code.total() < split_programs as f64, "{:?}", discovery.code);
    assert_ne!(discovery.behaviours[0].program.reads, Reads::Nothing);
}

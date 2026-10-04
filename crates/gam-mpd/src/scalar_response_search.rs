//! Optional proposal diagnostic over **every finite f32** scalar, including both zeros.
//!
//! For fixed, exact binary64 vectors this encloses the real-arithmetic objective
//! `max_t sum_j (a*v[t][j]-y[t][j])^2`. It does not model Artifact's rounded
//! derived-matrix multiplication, later forward rounding, Local or autonomous Run.
//! No fidelity acceptance or MatrixRule/global-program optimality follows.
//! Each cell is an inclusive interval of monotonically ordered f32 bit keys.
//! Basic arithmetic uses gam_math's directed ClosedInterval operations. Pruning
//! requires a cell lower bound >= an actually evaluated point's upper bound.
//! Budget exhaustion retains every unexplored cell and its lower bound.
use gam_math::score_opt::ClosedInterval as I;
use serde::Serialize;
use std::collections::BTreeSet;

const FIRST: u32 = 0x0080_0000;
const LAST: u32 = 0xff7f_ffff;
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Budget {
    pub max_cells: usize,
    pub max_evaluations: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub first_key: u32,
    pub last_key: u32,
    pub lower_bound: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct TestedPoint {
    pub amplitude_bits: u32,
    pub lower_bound: f64,
    pub upper_bound: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub budget: Budget,
    pub processed_cells: usize,
    pub evaluations_attempted: usize,
    pub lower_bound_excluded_keys: u64,
    pub total_finite_keys: u64,
    pub constant_response_equivalent_keys: u64,
    pub tested: Vec<TestedPoint>,
    pub unexplored: Vec<Cell>,
    pub lower_bound: f64,
    pub upper_bound: Option<f64>,
    pub gap_upper_bound: Option<f64>,
    pub best_amplitude_bits: Option<u32>,
    /// All finite keys were either evaluated or excluded by a valid lower bound.
    /// A positive arithmetic enclosure gap can remain even when this is true.
    pub complete_finite_inventory: bool,
    pub stop_reason: String,
    pub fixed_response_only: bool,
}
#[derive(Clone, Copy)]
struct Quadratic {
    a: I,
    b: I,
    c: I,
}
fn amplitude(key: u32) -> f32 {
    f32::from_bits(if key & 0x8000_0000 != 0 {
        key & 0x7fff_ffff
    } else {
        !key
    })
}
fn key(a: f32) -> u32 {
    let bits = a.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}
fn valid(x: I) -> bool {
    x.lo.is_finite() && x.hi.is_finite() && x.lo <= x.hi
}
fn product(x: f64, y: f64) -> Result<I, String> {
    let raw = x * y;
    if !raw.is_finite() || (x != 0.0 && y != 0.0 && raw == 0.0) {
        return Err("overflow or underflow in fixed-response arithmetic".into());
    }
    let p = I::point(x).mul(I::point(y));
    if valid(p) {
        Ok(p)
    } else {
        Err("nonfinite interval product".into())
    }
}
fn square(x: I) -> I {
    let low = if x.contains_zero() {
        0.0
    } else {
        x.lo.abs().min(x.hi.abs())
    };
    let high = x.lo.abs().max(x.hi.abs());
    I::new(
        I::point(low).mul(I::point(low)).lo.max(0.0),
        I::point(high).mul(I::point(high)).hi,
    )
}
fn quadratics(v: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Vec<Quadratic>, String> {
    v.iter()
        .zip(y)
        .map(|(v, y)| {
            let (mut a, mut b, mut c) = (I::point(0.0), I::point(0.0), I::point(0.0));
            for (&v, &y) in v.iter().zip(y) {
                if !v.is_finite() || !y.is_finite() {
                    return Err("nonfinite declared input".into());
                }
                a = a.add(product(v, v)?);
                b = b.add(product(v, y)?);
                c = c.add(product(y, y)?);
                if !valid(a) || !valid(b) || !valid(c) {
                    return Err("overflow in norm/coefficient accumulation".into());
                }
            }
            Ok(Quadratic { a, b, c })
        })
        .collect()
}
fn point(v: &[Vec<f64>], y: &[Vec<f64>], a: f32) -> Result<I, String> {
    let mut lower: f64 = 0.0;
    let mut upper: f64 = 0.0;
    for (v, y) in v.iter().zip(y) {
        let mut sum = I::point(0.0);
        for (&v, &y) in v.iter().zip(y) {
            let r = product(f64::from(a), v)?.sub(I::point(y));
            if !valid(r) {
                return Err("nonfinite response subtraction".into());
            }
            // Explicitly retain underflow as unresolved, rather than a zero score.
            for endpoint in [r.lo, r.hi] {
                product(endpoint, endpoint)?;
            }
            sum = sum.add(square(r));
            if !valid(sum) {
                return Err("nonfinite response norm".into());
            }
        }
        lower = lower.max(sum.lo.max(0.0));
        upper = upper.max(sum.hi);
    }
    Ok(I::new(lower, upper))
}
fn cell_bound(v: &[Vec<f64>], y: &[Vec<f64>], qs: &[Quadratic], first: u32, last: u32) -> f64 {
    let range = I::new(f64::from(amplitude(first)), f64::from(amplitude(last)));
    let mut bound: f64 = 0.0;
    for ((v, y), q) in v.iter().zip(y).zip(qs) {
        let mut direct = I::point(0.0);
        for (&v, &y) in v.iter().zip(y) {
            direct = direct.add(square(range.scale(v).sub(I::point(y))));
        }
        if valid(direct) {
            bound = bound.max(direct.lo.max(0.0));
        }
        // For each convex row quadratic, min over the cell occurs at an endpoint
        // when its derivative has a certified sign; otherwise its unrestricted
        // minimum C-B^2/A is a valid (possibly loose) cell lower bound.
        let row = if q.a.hi == 0.0 {
            q.c.lo
        } else if q.a.lo > 0.0 {
            let derivative = q.a.mul(range).sub(q.b);
            if valid(derivative) && (derivative.lo >= 0.0 || derivative.hi <= 0.0) {
                let at = I::point(if derivative.lo >= 0.0 {
                    range.lo
                } else {
                    range.hi
                });
                let value = q.a.mul(square(at)).sub(q.b.mul(at).scale(2.0)).add(q.c);
                if valid(value) { value.lo } else { 0.0 }
            } else {
                let value = q.c.sub(square(q.b).div_positive(q.a));
                if valid(value) { value.lo } else { 0.0 }
            }
        } else {
            0.0
        };
        bound = bound.max(row.max(0.0));
    }
    bound
}
/// Search a complete finite amplitude inventory under an explicit resource budget.
/// Numeric failures preserve an unresolved full-domain cell; no tolerance is used.
pub fn search(v: &[Vec<f64>], y: &[Vec<f64>], budget: Budget) -> Result<Report, String> {
    if v.is_empty() || v.len() != y.len() || v.iter().zip(y).any(|(v, y)| v.len() != y.len()) {
        return Err("nonempty matching response/target row shapes required".into());
    }
    let mut report = Report {
        budget,
        processed_cells: 0,
        evaluations_attempted: 0,
        lower_bound_excluded_keys: 0,
        total_finite_keys: u64::from(LAST) - u64::from(FIRST) + 1,
        constant_response_equivalent_keys: 0,
        tested: vec![],
        unexplored: vec![],
        lower_bound: 0.0,
        upper_bound: None,
        gap_upper_bound: None,
        best_amplitude_bits: None,
        complete_finite_inventory: false,
        stop_reason: String::new(),
        fixed_response_only: true,
    };
    let qs = match quadratics(v, y) {
        Ok(q) => q,
        Err(e) => {
            report.stop_reason = e;
            report.unexplored.push(Cell {
                first_key: FIRST,
                last_key: LAST,
                lower_bound: 0.0,
            });
            return Ok(report);
        }
    };
    if v.iter().flatten().all(|&x| x == 0.0) && budget.max_cells > 0 && budget.max_evaluations > 0 {
        report.evaluations_attempted += 1;
        match point(v, y, 0.0) {
            Ok(score) => {
                report.tested.push(TestedPoint {
                    amplitude_bits: 0.0_f32.to_bits(),
                    lower_bound: score.lo,
                    upper_bound: score.hi,
                });
                report.lower_bound = score.lo;
                report.upper_bound = Some(score.hi);
                report.gap_upper_bound =
                    Some(I::point(score.hi).sub(I::point(score.lo)).hi.max(0.0));
                report.best_amplitude_bits = Some(0.0_f32.to_bits());
                report.constant_response_equivalent_keys = report.total_finite_keys - 1;
                report.complete_finite_inventory = true;
                report.stop_reason =
                    "all responses exactly zero; objective independent of amplitude".into();
                return Ok(report);
            }
            Err(e) => {
                report.stop_reason = e;
                report.unexplored.push(Cell {
                    first_key: FIRST,
                    last_key: LAST,
                    lower_bound: 0.0,
                });
                return Ok(report);
            }
        }
    }
    let mut cells = vec![Cell {
        first_key: FIRST,
        last_key: LAST,
        lower_bound: cell_bound(v, y, &qs, FIRST, LAST),
    }];
    let mut seen = BTreeSet::new();
    let mut evaluated_lower = f64::INFINITY;
    // Least-squares is only a point-selection heuristic, never a pruning proof.
    let sum_a: f64 = qs.iter().map(|q| q.a.lo * 0.5 + q.a.hi * 0.5).sum();
    let sum_b: f64 = qs.iter().map(|q| q.b.lo * 0.5 + q.b.hi * 0.5).sum();
    let seed = if sum_a > 0.0 {
        (sum_b / sum_a) as f32
    } else {
        0.0
    };
    let mut preferred = seed.is_finite().then_some(key(seed));
    while !cells.is_empty() {
        if report
            .upper_bound
            .is_some_and(|u| cells.iter().all(|c| c.lower_bound >= u))
        {
            report.lower_bound_excluded_keys += cells
                .iter()
                .map(|c| u64::from(c.last_key) - u64::from(c.first_key) + 1)
                .sum::<u64>();
            cells.clear();
            break;
        }
        if report.processed_cells >= budget.max_cells
            || report.evaluations_attempted >= budget.max_evaluations
        {
            report.stop_reason = "explicit cell/evaluation budget exhausted".into();
            break;
        }
        let index = (0..cells.len())
            .min_by(|&a, &b| {
                cells[a]
                    .lower_bound
                    .total_cmp(&cells[b].lower_bound)
                    .then(cells[a].first_key.cmp(&cells[b].first_key))
            })
            .expect("nonempty frontier");
        let cell = cells.swap_remove(index);
        if report.upper_bound.is_some_and(|u| cell.lower_bound >= u) {
            report.lower_bound_excluded_keys +=
                u64::from(cell.last_key) - u64::from(cell.first_key) + 1;
            continue;
        }
        report.processed_cells += 1;
        let chosen = preferred
            .take()
            .filter(|&k| k >= cell.first_key && k <= cell.last_key)
            .unwrap_or(cell.first_key + (cell.last_key - cell.first_key) / 2);
        if seen.insert(chosen) {
            report.evaluations_attempted += 1;
            match point(v, y, amplitude(chosen)) {
                Ok(score) => {
                    evaluated_lower = evaluated_lower.min(score.lo);
                    report.tested.push(TestedPoint {
                        amplitude_bits: amplitude(chosen).to_bits(),
                        lower_bound: score.lo,
                        upper_bound: score.hi,
                    });
                    if report.upper_bound.is_none_or(|u| score.hi < u) {
                        report.upper_bound = Some(score.hi);
                        report.best_amplitude_bits = Some(amplitude(chosen).to_bits());
                    }
                }
                Err(e) => {
                    cells.push(cell);
                    report.stop_reason = e;
                    break;
                }
            }
        }
        // Remove only the tested key; the two disjoint children retain every
        // other finite key, including the distinct signed-zero representations.
        for (lo, hi) in [
            (cell.first_key, chosen.saturating_sub(1)),
            (chosen.saturating_add(1), cell.last_key),
        ] {
            if lo <= hi {
                let lb = cell_bound(v, y, &qs, lo, hi);
                if report.upper_bound.is_none_or(|u| lb < u) {
                    cells.push(Cell {
                        first_key: lo,
                        last_key: hi,
                        lower_bound: lb,
                    });
                } else {
                    report.lower_bound_excluded_keys += u64::from(hi) - u64::from(lo) + 1;
                }
            }
        }
    }
    report.complete_finite_inventory = cells.is_empty();
    if report.stop_reason.is_empty() {
        report.stop_reason = "all finite keys evaluated or lower-bound excluded".into();
    }
    report.lower_bound = cells
        .iter()
        .map(|c| c.lower_bound)
        .fold(evaluated_lower, f64::min)
        .min(report.upper_bound.unwrap_or(f64::INFINITY));
    if !report.lower_bound.is_finite() {
        report.lower_bound = 0.0;
    }
    report.gap_upper_bound = report
        .upper_bound
        .map(|u| I::point(u).sub(I::point(report.lower_bound)).hi.max(0.0));
    report.unexplored = cells;
    Ok(report)
}
#[cfg(test)]
#[path = "scalar_response_search_tests.rs"]
mod tests;

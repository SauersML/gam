//! #2280 (former #2909, folded back): the ball-cover census — exact pair and
//! triple intersection tests for the closed ambient balls against the sampled
//! membership predicate.
//!
//! The queued probe. The dismantlable-intersection producer was measured and
//! rejected (it fails every lattice-cell cover), and the follow-up model on
//! record is: each patch's closed ambient ball `B(c_i, r_i)` with `r_i` its
//! farthest member. Intersections of Euclidean balls are convex, so that cover
//! is good by construction, and the certificate it supports reduces to an exact
//! comparison — the membership nerve must equal the Čech nerve of the balls,
//! and a ball intersection no shared row witnesses is a typed refusal naming
//! the simplex. What was queued and not yet run: "a probe counting unwitnessed
//! ball pairs and triples on the planted zoo". This module is that probe.
//!
//! The intersection tests are exact, not sampled:
//!
//! * **Pairs.** The margin `f = min_x max(g_i, g_j)` with `g_l = ‖x−c_l‖ − r_l`
//!   is attained on the center segment at the crossing `s* = (D + r_i − r_j)/2`
//!   when it lies inside `[0, D]`, and at a center otherwise (a KKT point with
//!   two active constraints has its unit directions opposite, so it lies
//!   between the centers; one active constraint is the center itself). Both
//!   cases close to `f = max((D − r_i − r_j)/2, −min(r_i, r_j))`, and the pair
//!   intersects iff `f ≤ 0`, i.e. `D ≤ r_i + r_j`.
//!
//! * **Triples.** `f` is non-decreasing in every coordinate orthogonal to the
//!   centers' affine hull (`g_l(y + sn) = sqrt(g_l(y)² + s²) ≥ g_l(y)`), so the
//!   minimizer lies in the centers' plane, where KKT again confines it: one
//!   active constraint is a center; two active constraints put it on a center
//!   segment at the pair crossing; three active constraints make it an
//!   additively-weighted (Apollonius) circumcenter with `‖x−c_l‖ = ρ + r_l`
//!   sharing one `ρ`. Subtracting the squared equations pair by pair leaves two
//!   equations linear in `(x, ρ)`, so `x(ρ) = p + ρq` is affine and the single
//!   ball equation is a quadratic in `ρ` with at most two roots. The minimizer
//!   of `f` over the plane is the minimum of `f` over that finite candidate
//!   set — evaluated exactly, with no grid and no refinement parameter.
//!
//! The classification threshold is derived, not chosen: a simplex's own scale
//! is `scale = r_max + D_max` (its largest radius or center distance), and the
//! resolution is `16 · EPSILON · scale` — machine epsilon on the geometry's own
//! scale with a conservative factor for the chained roundings of the quadratic
//! solve. Margins inside that window are counted as boundary, never adjudicated.
//!
//! Containment is checked exactly instead of asserted: a shared member row
//! reproduces the builder's own squared-distance accumulation bit for bit, so
//! `sqrt` monotonicity gives `g_l(row) ≤ 0` with no roundtrip, and the exact
//! margin of a membership simplex must lie inside the resolution window of 0.

use super::*;
use crate::manifold::local_charts::LocalAtlasConfig;
use crate::manifold::tests_topology_fixtures::{
    circle, cylinder_strip, embedded_plane, mobius_strip, open_arc, sphere, swiss_roll,
    swiss_roll_with_height, torus, trefoil_knot,
};
use ndarray::ArrayView2;

/// One cover's closed ambient balls: `B(c_i, r_i)` with `r_i` the distance of
/// the center's farthest member, in the builder's own squared arithmetic.
struct BallCover {
    centers: Vec<Vec<f64>>,
    radii: Vec<f64>,
}

impl BallCover {
    fn build(atlas: &LocalAtlas, data: ArrayView2<'_, f64>) -> Self {
        let mut centers = Vec::with_capacity(atlas.chart_count());
        let mut radii = Vec::with_capacity(atlas.chart_count());
        for patch in atlas.patches() {
            let mut center = Vec::with_capacity(data.ncols());
            for column in 0..data.ncols() {
                center.push(data[[patch.center, column]]);
            }
            let radius2 = patch
                .members
                .iter()
                .map(|&row| {
                    (0..data.ncols())
                        .map(|column| (data[[row, column]] - data[[patch.center, column]]).powi(2))
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max);
            centers.push(center);
            radii.push(radius2.sqrt());
        }
        BallCover { centers, radii }
    }

    /// Squared ambient distance, accumulated per column exactly as the atlas
    /// builder accumulates patch radii.
    fn distance2(&self, a: &[f64], b: &[f64]) -> f64 {
        (0..b.len())
            .map(|column| (a[column] - b[column]).powi(2))
            .sum()
    }

    /// Exact pair margin: the minimum over `x` of `max(g_i, g_j)`. Negative
    /// means the closed balls intersect. Returns the margin and its argmin
    /// (the crossing point, or a center when one ball contains the other's).
    fn pair_margin(&self, i: usize, j: usize) -> (f64, Vec<f64>) {
        let distance = self.distance2(&self.centers[i], &self.centers[j]).sqrt();
        let crossing = ((distance - self.radii[i] - self.radii[j]) / 2.0)
            .max(-self.radii[i].min(self.radii[j]));
        if distance >= (self.radii[i] - self.radii[j]).abs() {
            // In-segment crossing at s* = (D + r_i − r_j)/2 from c_i.
            let s = ((distance + self.radii[i] - self.radii[j]) / 2.0).clamp(0.0, distance);
            let point = (0..self.centers[i].len())
                .map(|column| {
                    self.centers[i][column]
                        + (s / distance) * (self.centers[j][column] - self.centers[i][column])
                })
                .collect::<Vec<_>>();
            (crossing, point)
        } else {
            // One ball contains the other's center: the margin sits at the
            // contained ball's center.
            let contained = if self.radii[i] <= self.radii[j] { i } else { j };
            (crossing, self.centers[contained].clone())
        }
    }

    /// The resolution of one simplex's margins: machine epsilon on its own
    /// scale (largest radius or center distance) with a conservative factor
    /// for the quadratic solve's chained roundings.
    fn resolution(&self, simplex: &[usize]) -> f64 {
        let mut scale = simplex
            .iter()
            .map(|&l| self.radii[l])
            .fold(0.0_f64, f64::max);
        for a in 0..simplex.len() {
            for b in (a + 1)..simplex.len() {
                let distance = self
                    .distance2(&self.centers[simplex[a]], &self.centers[simplex[b]])
                    .sqrt();
                scale = scale.max(distance);
            }
        }
        16.0 * f64::EPSILON * scale
    }

    /// Exact triple margin: the minimum over the centers' plane of
    /// `max(g_i, g_j, g_k)`, attained on the finite KKT candidate set. Returns
    /// the margin and its argmin lifted back to ambient coordinates.
    fn triple_margin(&self, simplex: &[usize; 3]) -> (f64, Vec<f64>) {
        let [i, j, k] = *simplex;
        // Orthonormal basis of the centers' affine span: e1 along c_j − c_i,
        // e2 the in-plane component of c_k − c_i. Plane coordinates of a point
        // reproduce its true distances to the three centers.
        let dim = self.centers[i].len();
        let mut e1 = vec![0.0; dim];
        let mut w = vec![0.0; dim];
        let mut u = [0.0_f64; 3];
        let mut v = [0.0_f64; 3];
        let mut e1_norm2 = 0.0;
        for column in 0..dim {
            e1[column] = self.centers[j][column] - self.centers[i][column];
            w[column] = self.centers[k][column] - self.centers[i][column];
            e1_norm2 += e1[column] * e1[column];
        }
        let e1_norm = e1_norm2.sqrt();
        let dot = (0..dim).map(|column| e1[column] * w[column]).sum::<f64>();
        let mut e2 = vec![0.0; dim];
        let mut e2_norm2 = 0.0;
        for column in 0..dim {
            e2[column] = w[column] - (dot / e1_norm2) * e1[column];
            e2_norm2 += e2[column] * e2[column];
        }
        let e2_norm = e2_norm2.sqrt();
        let collinear = e2_norm <= self.resolution(simplex);
        for (position, l) in [i, j, k].into_iter().enumerate() {
            let dx = (0..dim)
                .map(|column| self.centers[l][column] - self.centers[i][column])
                .collect::<Vec<_>>();
            u[position] = (0..dim).map(|column| dx[column] * e1[column]).sum::<f64>() / e1_norm;
            v[position] = if collinear {
                0.0
            } else {
                (0..dim).map(|column| dx[column] * e2[column]).sum::<f64>() / e2_norm
            };
        }
        let r = [self.radii[i], self.radii[j], self.radii[k]];
        // f in plane coordinates: distances to the three centers from (x, y).
        let plane_f = |x: f64, y: f64| -> f64 {
            (0..3)
                .map(|l| ((x - u[l]) * (x - u[l]) + (y - v[l]) * (y - v[l])).sqrt() - r[l])
                .fold(f64::NEG_INFINITY, f64::max)
        };
        // Candidates 1 and 2: each center, and each center-segment crossing
        // s* = (D + r_a − r_b)/2 (clamped into the segment; outside it the
        // minimum on the segment sits at a center, already a candidate).
        let mut best = f64::INFINITY;
        let mut best_x = 0.0_f64;
        let mut best_y = 0.0_f64;
        let consider = |x: f64, y: f64, best: &mut f64, bx: &mut f64, by: &mut f64| {
            let value = plane_f(x, y);
            if value < *best {
                *best = value;
                *bx = x;
                *by = y;
            }
        };
        for l in 0..3 {
            consider(u[l], v[l], &mut best, &mut best_x, &mut best_y);
        }
        for a in 0..3 {
            for b in (a + 1)..3 {
                let dx = u[b] - u[a];
                let dy = v[b] - v[a];
                let distance = (dx * dx + dy * dy).sqrt();
                if distance <= self.resolution(simplex) {
                    continue;
                }
                let crossing = ((distance + r[a] - r[b]) / 2.0).clamp(0.0, distance);
                consider(
                    u[a] + (crossing / distance) * dx,
                    v[a] + (crossing / distance) * dy,
                    &mut best,
                    &mut best_x,
                    &mut best_y,
                );
            }
        }
        // Candidate 3: the Apollonius circumcenters. Subtracting the squared
        // equal-margin equations pair by pair (against chart 0) gives two
        // equations linear in (x, y, ρ):
        //   2 x (u_b − u_a) + 2 y (v_b − v_a) = 2 ρ (r_a − r_b) + (r_a² − r_b²)
        //     − (u_a² + v_a²) + (u_b² + v_b²)
        // so (x, y) = p + ρ q when the 2×2 system is invertible, and the chart-0
        // ball equation becomes a quadratic in ρ. Collinear centers skip this:
        // the whole configuration then lives on a line, where the piecewise
        // linear max attains its minimum at a crossing or a center.
        if !collinear {
            let c0 = [u[0], v[0]];
            let mut rows = [[0.0_f64; 3]; 2];
            let mut rhs = [0.0_f64; 2];
            for (row, b) in [1usize, 2usize].into_iter().enumerate() {
                rows[row][0] = 2.0 * (u[b] - u[0]);
                rows[row][1] = 2.0 * (v[b] - v[0]);
                rows[row][2] = -2.0 * (r[0] - r[b]);
                rhs[row] = (r[0] * r[0] - r[b] * r[b]) - (u[0] * u[0] + v[0] * v[0])
                    + (u[b] * u[b] + v[b] * v[b]);
            }
            let determinant = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0];
            if determinant.abs() > self.resolution(simplex) {
                // Particular solution at ρ = 0 and the affine direction in ρ.
                let px = (rhs[0] * rows[1][1] - rows[0][1] * rhs[1]) / determinant;
                let py = (rows[0][0] * rhs[1] - rhs[0] * rows[1][0]) / determinant;
                let rhos = [-rows[0][2], -rows[1][2]];
                let qx = (rhos[0] * rows[1][1] - rows[0][1] * rhos[1]) / determinant;
                let qy = (rows[0][0] * rhos[1] - rhos[0] * rows[1][0]) / determinant;
                // ‖p + ρ q − c0‖² = (ρ + r_0)²:
                //   (qx² + qy² − 1) ρ² + 2 ((p−c0)·q − r_0) ρ + ‖p − c0‖² − r_0² = 0.
                let mx = px - c0[0];
                let my = py - c0[1];
                let qa = qx * qx + qy * qy - 1.0;
                let qb = 2.0 * (mx * qx + my * qy - r[0]);
                let qc = mx * mx + my * my - r[0] * r[0];
                let roots = solve_quadratic(qa, qb, qc);
                for rho in roots {
                    consider(
                        px + rho * qx,
                        py + rho * qy,
                        &mut best,
                        &mut best_x,
                        &mut best_y,
                    );
                }
            }
        }
        // Lift the argmin back to ambient coordinates through the plane basis.
        let argmin = (0..dim)
            .map(|column| {
                self.centers[i][column]
                    + best_x * e1[column] / e1_norm
                    + best_y * e2[column] / e2_norm
            })
            .collect::<Vec<_>>();
        (best, argmin)
    }
}

/// Real roots of `a t² + b t + c = 0` under the conservative determinant
/// guard; a near-vanishing leading coefficient falls back to the linear root.
fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
    let resolution = 16.0 * f64::EPSILON * scale;
    if a.abs() <= resolution {
        if b.abs() <= resolution {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let root = discriminant.sqrt();
    // The numerically stable pairing: q = -(b + sign(b)·√disc)/2, roots
    // q/a and c/q.
    let q = -0.5 * (b + root * b.signum());
    if q.abs() <= resolution {
        return vec![-0.5 * b / a];
    }
    vec![q / a, c / q]
}

/// One fixture's census line.
struct CensusLine {
    label: &'static str,
    charts: usize,
    /// Ball-intersecting pairs no shared row witnesses.
    pairs_unwitnessed: usize,
    /// Pair margins inside the resolution window (never adjudicated).
    pairs_boundary: usize,
    /// Ball-intersecting triples no shared row witnesses.
    triples_unwitnessed: usize,
    /// Triple margins inside the resolution window.
    triples_boundary: usize,
    /// Membership simplices whose exact margin exceeds the resolution window
    /// (must stay zero: a shared row lies in every ball by construction).
    containment_violations: usize,
    /// The most negative unwitnessed margin seen (how deep the refusal cuts).
    deepest_unwitnessed_margin: f64,
    /// Unwitnessed intersections whose argmin sits farther from every row than
    /// 3× the fixture's median nearest-neighbor spacing: off-manifold.
    pairs_off_sheet: usize,
    triples_off_sheet: usize,
    /// The fixture's median nearest-neighbor spacing (the classification
    /// scale, reported with the counts).
    spacing: f64,
}

/// Run the census on one fixture at the builder's own derived configuration.
fn census(label: &'static str, data: ArrayView2<'_, f64>, d: usize) -> CensusLine {
    let config = LocalAtlasConfig::balanced(data.nrows(), d);
    let atlas = LocalAtlas::build(data.view(), config).expect("fixture atlas must build");
    let balls = BallCover::build(&atlas, data.view());
    let count = atlas.chart_count();
    let patches: Vec<&Vec<usize>> = atlas.patches().iter().map(|p| &p.members).collect();

    let mut pairs_unwitnessed = 0;
    let mut pairs_boundary = 0;
    let mut triples_unwitnessed = 0;
    let mut triples_boundary = 0;
    let mut containment_violations = 0;
    let mut deepest = f64::INFINITY;
    // Off-sheet classification: how deep inside the intersection the argmin
    // sits versus how far the argmin sits from any sampled row. An argmin
    // farther from every row than a multiple of the fixture's own median
    // nearest-row spacing cannot be an under-sampled on-sheet point — the
    // intersection lives off the manifold (across a fold or gap), and no
    // refinement of the sample would ever witness it.
    let mut pairs_off_sheet = 0;
    let mut triples_off_sheet = 0;
    let mut unwitnessed_pairs: Vec<(usize, usize)> = Vec::new();
    let mut unwitnessed_triples: Vec<(usize, usize, usize)> = Vec::new();

    // The fixture's own sampling scale: the median nearest-neighbor distance
    // among its rows.
    let mut nn = Vec::with_capacity(data.nrows().min(400));
    for row in 0..data.nrows() {
        let mut best2 = f64::INFINITY;
        for other in 0..data.nrows() {
            if other == row {
                continue;
            }
            let d2 = (0..data.ncols())
                .map(|column| (data[[row, column]] - data[[other, column]]).powi(2))
                .sum::<f64>();
            if d2 < best2 {
                best2 = d2;
            }
        }
        nn.push(best2.sqrt());
    }
    nn.sort_by(|a, b| a.partial_cmp(b).expect("finite distances"));
    let spacing = nn[nn.len() / 2];
    let nearest_row = |point: &[f64]| -> f64 {
        (0..data.nrows())
            .map(|row| {
                (0..data.ncols())
                    .map(|column| (point[column] - data[[row, column]]).powi(2))
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
            .sqrt()
    };

    for a in 0..count {
        for b in (a + 1)..count {
            let simplex = [a, b];
            let (margin, argmin) = balls.pair_margin(a, b);
            let resolution = balls.resolution(&simplex);
            let witnessed = !sorted_intersection(patches[a], patches[b]).is_empty();
            if witnessed {
                if margin > resolution {
                    containment_violations += 1;
                }
            } else if margin < -resolution {
                pairs_unwitnessed += 1;
                deepest = deepest.min(margin);
                if nearest_row(&argmin) > 3.0 * spacing {
                    pairs_off_sheet += 1;
                }
                if unwitnessed_pairs.len() < 8 {
                    unwitnessed_pairs.push((a, b));
                }
            } else if margin <= resolution {
                pairs_boundary += 1;
            }
        }
    }

    for a in 0..count {
        for b in (a + 1)..count {
            for c in (b + 1)..count {
                let simplex = [a, b, c];
                let (margin, argmin) = balls.triple_margin(&simplex);
                let resolution = balls.resolution(&simplex);
                let mut shared = sorted_intersection(patches[a], patches[b]);
                shared.retain(|&row| patches[c].binary_search(&row).is_ok());
                if !shared.is_empty() {
                    if margin > resolution {
                        containment_violations += 1;
                    }
                } else if margin < -resolution {
                    triples_unwitnessed += 1;
                    deepest = deepest.min(margin);
                    if nearest_row(&argmin) > 3.0 * spacing {
                        triples_off_sheet += 1;
                    }
                    if unwitnessed_triples.len() < 8 {
                        unwitnessed_triples.push((a, b, c));
                    }
                } else if margin <= resolution {
                    triples_boundary += 1;
                }
            }
        }
    }

    eprintln!(
        "#2280 census {label}: charts={count} pairs_unwitnessed_nonempty={pairs_unwitnessed} \
         pairs_boundary={pairs_boundary} triples_unwitnessed_nonempty={triples_unwitnessed} \
         triples_boundary={triples_boundary} containment_violations={containment_violations} \
         deepest_unwitnessed_margin={deepest:.6} pairs_off_sheet={pairs_off_sheet} \
         triples_off_sheet={triples_off_sheet} spacing={spacing:.6}",
    );
    if !unwitnessed_pairs.is_empty() {
        eprintln!("#2280 census {label}: first unwitnessed pairs {unwitnessed_pairs:?}");
    }
    if !unwitnessed_triples.is_empty() {
        eprintln!("#2280 census {label}: first unwitnessed triples {unwitnessed_triples:?}");
    }

    CensusLine {
        label,
        charts: count,
        pairs_unwitnessed,
        pairs_boundary,
        triples_unwitnessed,
        triples_boundary,
        containment_violations,
        deepest_unwitnessed_margin: deepest,
        pairs_off_sheet,
        triples_off_sheet,
        spacing,
    }
}

#[test]
fn ball_census_runs_the_planted_zoo_2280() {
    let lines = vec![
        census("swiss_roll_80x16", swiss_roll(80, 16).view(), 2),
        census(
            "swiss_roll_60x16_h10",
            swiss_roll_with_height(60, 16, 10.0).view(),
            2,
        ),
        census(
            "swiss_roll_80x16_h10",
            swiss_roll_with_height(80, 16, 10.0).view(),
            2,
        ),
        census(
            "swiss_roll_80x30_h20",
            swiss_roll_with_height(80, 30, 20.0).view(),
            2,
        ),
        census("torus_60x26", torus(60, 26, 2.0, 0.8).view(), 2),
        census("mobius_60x14", mobius_strip(60, 14).view(), 2),
        census("sphere_400", sphere(400).view(), 2),
        census("sphere_900", sphere(900).view(), 2),
        census("cylinder_40x10", cylinder_strip(40, 10).view(), 2),
        census("cylinder_60x14", cylinder_strip(60, 14).view(), 2),
        census("circle_200", circle(200, 2.0).view(), 1),
        census("circle_400", circle(400, 2.0).view(), 1),
        census("open_arc_200", open_arc(200, 2.0).view(), 1),
        census("embedded_plane_24x24", embedded_plane(24, 24).view(), 2),
        census("trefoil_600", trefoil_knot(600, 1.0).view(), 1),
    ];
    // Structural pins. The equality certificate (membership nerve = ball
    // Čech nerve) is satisfiable only where unwitnessed == 0: exactly the two
    // densely-covered one-manifolds. The trefoil's refusals are ALL
    // off-sheet — ambient bridges between knot strands, which no sampling
    // density can witness — so the ball model is structurally wrong there,
    // not merely under-sampled.
    let clean: Vec<_> = lines
        .iter()
        .filter(|l| l.pairs_unwitnessed == 0 && l.triples_unwitnessed == 0)
        .map(|l| l.label)
        .collect();
    assert_eq!(
        clean,
        vec!["circle_200", "open_arc_200"],
        "only the dense one-manifold covers satisfy the ball-equality certificate"
    );
    let trefoil = lines
        .iter()
        .find(|l| l.label == "trefoil_600")
        .expect("trefoil census line");
    assert!(
        trefoil.pairs_unwitnessed > 0 && trefoil.pairs_off_sheet == trefoil.pairs_unwitnessed,
        "the trefoil's unwitnessed ball pairs must all sit off-sheet"
    );
    assert!(
        trefoil.triples_unwitnessed > 0 && trefoil.triples_off_sheet == trefoil.triples_unwitnessed,
        "the trefoil's unwitnessed ball triples must all sit off-sheet"
    );

    for line in &lines {
        eprintln!(
            "#2280 census pinned {label}: charts={charts} pairs_unwitnessed={pairs_unwitnessed} \
             triples_unwitnessed={triples_unwitnessed} boundary=({pairs_boundary}, \
             {triples_boundary}) deepest={d:.6} off_sheet={off_sheet:?} spacing={spacing:.4}",
            label = line.label,
            charts = line.charts,
            pairs_unwitnessed = line.pairs_unwitnessed,
            triples_unwitnessed = line.triples_unwitnessed,
            pairs_boundary = line.pairs_boundary,
            triples_boundary = line.triples_boundary,
            d = line.deepest_unwitnessed_margin,
            off_sheet = (line.pairs_off_sheet, line.triples_off_sheet),
            spacing = line.spacing,
        );
        assert_eq!(
            line.containment_violations, 0,
            "{}: a membership simplex outside the ball intersection contradicts containment",
            line.label
        );
    }
}

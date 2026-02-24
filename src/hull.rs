use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use serde::{Deserialize, Serialize};

/// Error type for hull building and projection.
#[derive(thiserror::Error, Debug)]
pub enum HullError {
    #[error(
        "Input data must have at least d+1 points to define a hull. Got {0} points for dimension {1}."
    )]
    InsufficientPoints(usize, usize),
}

/// A peeled convex hull represented as an intersection of half-spaces a^T x <= b.
/// Facet normals `a` are unit-length direction vectors used to generate supporting halfspaces
/// after iterative peeling. This is a robust, outlier-insensitive boundary representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeeledHull {
    /// Each facet as (normal, offset). For any in-domain x: a^T x <= b for all facets.
    pub facets: Vec<(Array1<f64>, f64)>,
    /// Dimensionality of the space (number of predictors)
    pub dim: usize,
}

impl PeeledHull {
    /// Projects points in place onto the hull if needed. Returns the count of projected points.
    pub fn project_in_place(&self, mut points: ArrayViewMut2<'_, f64>) -> usize {
        if points.nrows() == 0 {
            return 0;
        }

        points
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .map(|mut row| {
                let view = row.view();
                if self.is_inside(view) {
                    0usize
                } else {
                    let proj = self.project_point(view);
                    row.assign(&proj);
                    1usize
                }
            })
            .sum()
    }

    /// Projects points onto the hull if needed. Returns corrected points and count projected.
    pub fn project_if_needed(&self, points: ArrayView2<f64>) -> (Array2<f64>, usize) {
        let d = points.ncols();
        assert_eq!(
            d, self.dim,
            "Dimension mismatch in PeeledHull::project_if_needed"
        );

        let mut out = points.to_owned();
        let projected = self.project_in_place(out.view_mut());
        (out, projected)
    }

    /// Fast in-domain test: a_i^T x <= b_i for all facets.
    pub fn is_inside(&self, x: ArrayView1<f64>) -> bool {
        for (a, b) in &self.facets {
            let s = a.dot(&x);
            if s > *b + 1e-12 {
                return false;
            }
        }
        true
    }

    /// Compute the signed distance from a point to the peeled hull boundary.
    ///
    /// Convention:
    /// - Negative inside: exact distance to the nearest boundary facet.
    /// - Positive outside: exact Euclidean distance to the polytope (via projection).
    /// - Zero on the boundary.
    pub fn signed_distance(&self, x: ArrayView1<f64>) -> f64 {
        if self.is_inside(x) {
            // Inside: distance to boundary is the minimum slack over facets.
            // Facet normals are constructed unit-length, so slack equals Euclidean distance.
            let mut min_slack = f64::INFINITY;
            for (a, b) in &self.facets {
                let slack = *b - a.dot(&x);
                if slack < min_slack {
                    min_slack = slack;
                }
            }
            // Numerical safety: never return a negative slack for inside points
            -min_slack.max(0.0)
        } else {
            // Outside: use Dykstra projection onto the feasible polytope
            let z = self.project_point(x);
            let diff = &x.to_owned() - &z;

            diff.mapv(|v| v * v).sum().sqrt()
        }
    }

    /// Vectorized signed distance for a batch of points (row-wise).
    pub fn signed_distance_many(&self, points: ArrayView2<f64>) -> Array1<f64> {
        let n = points.nrows();
        let mut out = Array1::zeros(n);
        if n == 0 {
            return out;
        }

        out.as_slice_mut()
            .expect("contiguous slice")
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                *slot = self.signed_distance(points.row(i));
            });
        out
    }

    /// Compute signed distances and corresponding projections in a single pass.
    /// Returns (signed_distances, projected_points).
    pub fn signed_distance_and_project_many(
        &self,
        points: ArrayView2<f64>,
    ) -> (Array1<f64>, Array2<f64>) {
        let n = points.nrows();
        let d = points.ncols();
        let mut dist = Array1::zeros(n);
        let mut proj = Array2::zeros((n, d));
        if n == 0 {
            return (dist, proj);
        }

        let results: Vec<(f64, Vec<f64>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point_row = points.row(i);
                if self.is_inside(point_row) {
                    let mut min_slack = f64::INFINITY;
                    for (a, b) in &self.facets {
                        let slack = *b - a.dot(&point_row);
                        if slack < min_slack {
                            min_slack = slack;
                        }
                    }
                    let dist_val = -min_slack.max(0.0);
                    let proj_vec = point_row.to_vec();
                    (dist_val, proj_vec)
                } else {
                    let zi = self.project_point(point_row);
                    let diff = point_row.to_owned() - &zi;
                    let dist_val = diff.mapv(|v| v * v).sum().sqrt();
                    let proj_vec = zi.to_vec();
                    (dist_val, proj_vec)
                }
            })
            .collect();

        for (i, (dist_val, proj_vec)) in results.into_iter().enumerate() {
            dist[i] = dist_val;
            proj.row_mut(i).assign(&ArrayView1::from(&proj_vec));
        }

        (dist, proj)
    }

    /// Project a single point onto the polytope using Dykstra's algorithm for halfspaces.
    fn project_point(&self, y: ArrayView1<f64>) -> Array1<f64> {
        let d = self.dim;
        let m = self.facets.len();
        let max_cycles = 200; // cycles over all constraints
        let tol = 1e-8;

        // Dykstra variables
        let mut x = y.to_owned();
        let mut p_corr: Vec<Array1<f64>> = (0..m).map(|_| Array1::zeros(d)).collect();

        for _ in 0..max_cycles {
            let x_prev = x.clone();
            for (i, (a, b)) in self.facets.iter().enumerate() {
                // y_i = x + p_i
                let mut y_i = x.clone();
                y_i += &p_corr[i];

                // Project y_i onto halfspace H_i: a^T z <= b
                let a_tb = a.dot(&y_i) - *b;
                let a_norm2 = a.dot(a).max(1e-16);
                if a_tb > 0.0 {
                    // Outside; move along normal inward
                    let alpha = a_tb / a_norm2;
                    let z = &y_i - &(a * alpha);
                    // Update correction and current x
                    p_corr[i] = &y_i - &z;
                    x = z;
                } else {
                    // Inside; projection is itself
                    p_corr[i].fill(0.0);
                    x = y_i;
                }
            }

            // Convergence check
            let diff = (&x - &x_prev).mapv(|v| v.abs()).sum();
            if diff < tol {
                return x;
            }
        }

        // If not converged, return last iterate (still feasible or near-feasible)
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn unit_square_hull() -> PeeledHull {
        // 0 <= x <= 1, 0 <= y <= 1
        PeeledHull {
            facets: vec![
                (array![1.0, 0.0], 1.0),  // x <= 1
                (array![-1.0, 0.0], 0.0), // -x <= 0 -> x >= 0
                (array![0.0, 1.0], 1.0),  // y <= 1
                (array![0.0, -1.0], 0.0), // -y <= 0 -> y >= 0
            ],
            dim: 2,
        }
    }

    #[test]
    fn test_is_inside_unit_square() {
        let h = unit_square_hull();
        assert!(h.is_inside(array![0.5, 0.5].view())); // inside
        assert!(h.is_inside(array![1.0, 0.5].view())); // on edge
        assert!(h.is_inside(array![0.0, 0.0].view())); // corner
        assert!(!h.is_inside(array![1.1, 0.5].view())); // outside +x
        assert!(!h.is_inside(array![-0.1, 0.5].view())); // outside -x
        assert!(!h.is_inside(array![0.5, 1.1].view())); // outside +y
        assert!(!h.is_inside(array![0.5, -0.1].view())); // outside -y
    }

    #[test]
    fn test_project_point_unit_square() {
        let h = unit_square_hull();
        // Inside point stays unchanged
        let p_in = array![0.5, 0.5];
        let proj_in = h.project_point(p_in.view());
        assert!((&proj_in - &p_in).mapv(|v| v.abs()).sum() < 1e-12);

        // Project onto a face
        let p_face = array![1.5, 0.5];
        let proj_face = h.project_point(p_face.view());
        assert!((&proj_face - &array![1.0, 0.5]).mapv(|v| v.abs()).sum() < 1e-8);

        // Project onto a corner
        let p_corner = array![1.5, -0.5];
        let proj_corner = h.project_point(p_corner.view());
        assert!((&proj_corner - &array![1.0, 0.0]).mapv(|v| v.abs()).sum() < 1e-6);
    }

    #[test]
    fn test_project_if_needed_with_outliers() {
        // Build a small 2D clustered training set in [-1, 1]^2
        let mut pts = Vec::new();
        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                pts.push([x, y]);
            }
        }
        let data = ndarray::Array2::from(pts);
        let hull = build_peeled_hull(&data, 2).expect("hull build failed");

        // Mix of inside and outliers
        let test = ndarray::arr2(&[[0.2, 0.2], [2.0, 2.0], [-2.0, -2.0]]);
        let (corrected, num_proj) = hull.project_if_needed(test.view());
        assert_eq!(num_proj, 2);
        // First point unchanged
        assert!(
            (corrected.row(0).to_owned() - test.row(0).to_owned())
                .mapv(|v| v.abs())
                .sum()
                < 1e-12
        );
        // All corrected points must be inside
        for i in 0..corrected.nrows() {
            assert!(hull.is_inside(corrected.row(i)));
        }
    }

    #[test]
    fn test_signed_distance_unit_square() {
        let h = unit_square_hull();
        // Inside center: nearest boundary at distance 0.5 (negative inside)
        let d_center = h.signed_distance(array![0.5, 0.5].view());
        assert!((d_center + 0.5).abs() < 1e-12);

        // Inside near left edge: distance ~0.2 (negative)
        let d_inside = h.signed_distance(array![0.2, 0.8].view());
        assert!((d_inside + 0.2).abs() < 1e-12);

        // On edge: exactly zero (treat as on/inside)
        let d_edge = h.signed_distance(array![1.0, 0.3].view());
        assert!(d_edge.abs() < 1e-12);

        // Outside along +x: distance 0.5
        let d_out_x = h.signed_distance(array![1.5, 0.5].view());
        assert!((d_out_x - 0.5).abs() < 1e-8);

        // Outside towards corner: distance sqrt(0.5^2 + 0.5^2)
        let d_out_corner = h.signed_distance(array![1.5, -0.5].view());
        assert!((d_out_corner - (0.5f64.hypot(0.5))).abs() < 1e-6);
    }
}

/// Builds a peeled hull from data using iterative peeling with directional supports.
/// This uses a fixed bank of direction vectors to compute supporting halfspaces and
/// approximate the convex hull robustly without external dependencies.
pub fn build_peeled_hull(data: &Array2<f64>, peels: usize) -> Result<PeeledHull, HullError> {
    let n = data.nrows();
    let d = data.ncols();
    if n < d + 1 {
        return Err(HullError::InsufficientPoints(n, d));
    }

    // Copy working set of points
    let mut current = data.clone();

    // Deterministic direction set: ±e_i plus a set of pseudo-random unit vectors
    let directions = generate_directions(d, 8 * d); // total ≈ 10d with axes

    // Iterative peeling
    for _ in 0..peels.max(1) {
        if current.nrows() < d + 1 {
            break;
        }
        let verts = extreme_point_indices(&current, &directions);
        if verts.is_empty() {
            break;
        }
        // Early stop if removing too many points or collapsing dimension
        if verts.len() as f64 > 0.25 * current.nrows() as f64
            || current.nrows() - verts.len() < d + 1
        {
            break;
        }
        // Remove rows at `verts` from current.
        // Build an O(n + |verts|) boolean drop mask instead of repeated contains() scans.
        let mut drop_mask = vec![false; current.nrows()];
        for &idx in &verts {
            if idx < drop_mask.len() {
                drop_mask[idx] = true;
            }
        }
        let keep_count = drop_mask.iter().filter(|&&drop| !drop).count();
        let mut next = Array2::zeros((keep_count, d));
        let mut r = 0;
        for (i, &drop) in drop_mask.iter().enumerate() {
            if !drop {
                next.row_mut(r).assign(&current.row(i));
                r += 1;
            }
        }
        current = next;
    }

    // Final facets from remaining core points: support values along same directions
    let mut facets: Vec<(Array1<f64>, f64)> = Vec::with_capacity(directions.len());
    for a in directions {
        // Ensure unit-length for numerical stability
        let norm = a.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let a_unit = a.mapv(|v| v / norm);
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..current.nrows() {
            let s = a_unit.dot(&current.row(i));
            if s > max_val {
                max_val = s;
            }
        }
        facets.push((a_unit, max_val));
    }

    Ok(PeeledHull { facets, dim: d })
}

/// Generate a deterministic bank of direction vectors for dimension d.
/// Includes ±standard basis and additional quasi-random directions from an LCG.
fn generate_directions(d: usize, extra: usize) -> Vec<Array1<f64>> {
    let mut dirs: Vec<Array1<f64>> = Vec::new();
    // ± e_i
    for i in 0..d {
        let mut e = Array1::zeros(d);
        e[i] = 1.0;
        dirs.push(e.clone());
        let mut en = Array1::zeros(d);
        en[i] = -1.0;
        dirs.push(en);
    }
    // LCG-based unit vectors
    let mut seed: u64 = 0xDEADBEEFCAFEBABE;
    let a: u64 = 6364136223846793005;
    let c: u64 = 1;
    let m: u64 = 1u64 << 63;
    for k in 0..extra {
        seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
        // Fill with uniform(-0.5, 0.5)
        let mut v = Array1::zeros(d);
        for j in 0..d {
            seed = (a.wrapping_mul(seed).wrapping_add(c)) % m;
            let u = (seed as f64) / (m as f64); // [0,1)
            v[j] = u - 0.5;
        }
        // Normalize; if degenerate, fall back to an axis
        let norm = v.mapv(|x| x * x).sum().sqrt();
        if norm > 1e-12 {
            dirs.push(v.mapv(|x| x / norm));
        } else {
            let mut e = Array1::zeros(d);
            e[k % d] = 1.0;
            dirs.push(e);
        }
    }
    dirs
}

/// Identify indices of extreme points for the current point cloud using support
/// along a bank of directions. Returns unique indices across all directions.
fn extreme_point_indices(points: &Array2<f64>, directions: &[Array1<f64>]) -> Vec<usize> {
    let n = points.nrows();
    let mut mark = vec![false; n];
    let mut unique = 0usize;
    for a in directions {
        let mut argmax = 0usize;
        let mut best = f64::NEG_INFINITY;
        for i in 0..n {
            let s = a.dot(&points.row(i));
            if s > best {
                best = s;
                argmax = i;
            }
        }
        if !mark[argmax] {
            mark[argmax] = true;
            unique += 1;
        }
    }
    let mut out = Vec::with_capacity(unique);
    for (i, &is_marked) in mark.iter().enumerate() {
        if is_marked {
            out.push(i);
        }
    }
    out
}

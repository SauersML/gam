use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use serde::{Deserialize, Serialize};

/// Slack tolerance for the half-space membership test `aᵀx ≤ b`. Points within
/// this distance of a facet (normals are unit-length, so slack is Euclidean
/// distance) are treated as on-boundary / inside, absorbing round-off in the
/// dot product.
const MEMBERSHIP_SLACK_TOL: f64 = 1e-12;

/// Maximum number of Dykstra cycles over all facet constraints before the
/// projection returns its last (near-feasible) iterate.
const DYKSTRA_MAX_CYCLES: usize = 200;

/// Convergence tolerance on the per-cycle L¹ iterate change for Dykstra
/// projection.
const DYKSTRA_TOL: f64 = 1e-8;

/// Floor on the squared facet-normal magnitude used when normalising the
/// projection step, guarding against a degenerate (near-zero) normal.
const FACET_NORM2_FLOOR: f64 = 1e-16;

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
    pub(crate) fn project_in_place(&self, mut points: ArrayViewMut2<'_, f64>) -> usize {
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
            if s > *b + MEMBERSHIP_SLACK_TOL {
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
            let diff = &z - &x;

            diff.mapv(|v| v * v).sum().sqrt()
        }
    }

    /// Project a single point onto the polytope using Dykstra's algorithm for halfspaces.
    fn project_point(&self, y: ArrayView1<f64>) -> Array1<f64> {
        let d = self.dim;
        let m = self.facets.len();
        let max_cycles = DYKSTRA_MAX_CYCLES;
        let tol = DYKSTRA_TOL;

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
                let a_norm2 = a.dot(a).max(FACET_NORM2_FLOOR);
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

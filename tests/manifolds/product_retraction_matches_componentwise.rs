use gam::{CircleManifold, EuclideanManifold, ProductManifold, RiemannianManifold};
use ndarray::{arr1, s};

#[test]
fn product_retraction_matches_componentwise() {
    let product = ProductManifold::new(vec![
        Box::new(CircleManifold::new()),
        Box::new(EuclideanManifold::new(2)),
    ]);

    let x = arr1(&[0.25, 1.0, -2.0]);
    let xi = arr1(&[0.7, 0.3, -0.4]);

    let y_product = product
        .retract(x.view(), xi.view())
        .expect("product retraction should succeed");

    let circle = CircleManifold::new();
    let euclidean = EuclideanManifold::new(2);
    let y0 = circle
        .retract(x.slice(s![0..1]), xi.slice(s![0..1]))
        .expect("circle retraction should succeed");
    let y1 = euclidean
        .retract(x.slice(s![1..3]), xi.slice(s![1..3]))
        .expect("euclidean retraction should succeed");

    let y_concat = arr1(&[y0[0], y1[0], y1[1]]);
    let diff = (&y_product - &y_concat).mapv(f64::abs).sum();
    assert!(
        diff < 1.0e-12,
        "Product-manifold retraction should match retraction applied independently to each component"
    );
}

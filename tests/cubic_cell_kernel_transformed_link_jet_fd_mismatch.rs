use gam::families::cubic_cell_kernel::{LocalSpanCubic, transformed_link_cubic};

fn eval(span: LocalSpanCubic, x: f64) -> f64 {
    span.evaluate(x)
}

#[test]
fn cubic_cell_kernel_transformed_link_second_and_third_derivatives_match_fd() {
    let span = LocalSpanCubic {
        left: -0.7,
        right: 2.0,
        c0: 0.21,
        c1: -0.43,
        c2: 0.37,
        c3: -0.19,
    };
    let a = 0.38;
    let b = -1.17;

    let (_d0, d1, d2, d3) = transformed_link_cubic(span, a, b);

    for &h in &[1e-3, 1e-5, 1e-7] {
        let f = |z: f64| eval(span, a + b * z);
        let fm2 = f(-2.0 * h);
        let fm1 = f(-h);
        let f0 = f(0.0);
        let fp1 = f(h);
        let fp2 = f(2.0 * h);

        let fd1 = (fp1 - fm1) / (2.0 * h);
        let fd2 = (fp1 - 2.0 * f0 + fm1) / (h * h);
        let fd3 = (fm2 - 2.0 * fm1 + 2.0 * fp1 - fp2) / (2.0 * h * h * h);

        assert!((d1 - fd1).abs() < 1e-6, "h={h}, d1={d1}, fd1={fd1}");
        assert!((d2 - fd2).abs() < 1e-5, "h={h}, d2={d2}, fd2={fd2}");
        assert!((d3 - fd3).abs() < 1e-4, "h={h}, d3={d3}, fd3={fd3}");
    }
}

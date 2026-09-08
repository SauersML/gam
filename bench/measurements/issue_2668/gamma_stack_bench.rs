//! Fixed-input performance and bitwise-output receipt for the gamma derivative
//! stack. Compile against each library with the same rustc flags, then compare
//! the fingerprints before interpreting the timing ratio.

use std::hint::black_box;
use std::time::Instant;

fn main() {
    let inputs = [
        1.0e-8, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 19.9, 20.0, 50.0, 1.0e4, 1.0e8,
    ];
    let rounds = 10_000;
    let start = Instant::now();
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..rounds {
        for x in inputs {
            let stack = gam_math::jet_tower::ln_gamma_derivative_stack(black_box(x));
            for value in black_box(stack) {
                assert!(value.is_finite(), "non-finite gamma derivative at {x}");
                fingerprint = (fingerprint ^ value.to_bits()).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    let elapsed = start.elapsed();
    println!(
        "evaluations={} elapsed_seconds={:.6} ns_per_stack={:.1} fingerprint={fingerprint:016x}",
        rounds * inputs.len(),
        elapsed.as_secs_f64(),
        elapsed.as_nanos() as f64 / (rounds * inputs.len()) as f64,
    );
}

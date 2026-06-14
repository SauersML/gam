
// Unit tests are crucial for a mathematical library like this.
#[cfg(test)]
mod tests {
include!("../../../tests/src_modules/basis_radial_periodic_thinplate_tests.rs");
include!("../../../tests/src_modules/basis_duchon_matern_jet_derivative_tests.rs");
}

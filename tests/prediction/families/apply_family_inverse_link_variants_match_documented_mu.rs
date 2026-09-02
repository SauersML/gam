use gam::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, MixtureLinkState,
    ResponseFamily, SasLinkState,
};
use gam_predict::predict_gam;
use ndarray::{arr1, arr2};

fn std_norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}


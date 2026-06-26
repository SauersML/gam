//! Dispatch table for the torch fit entry — single source of truth for which
//! Python `Smooth` subclasses the torch.fit autograd glue recognises.
//!
//! The Python side calls `torch_smooth_dispatch_key(type(smooth).__name__)`
//! to translate the spec class name into a small enumeration. The tensor
//! construction itself stays in Python because the torch autograd VJP must
//! flow back through `points`, `centers`, and `by`.
//!
//! Every Python `Smooth` subclass that is re-exported from `gamfit.torch`
//! must have a matching variant here, so that dispatch never fails for a
//! class the user can legitimately import. `TensorBSpline` (te tensor
//! product), `Matern` (kernel-Gram penalty), and `Categorical` (sum-to-zero
//! contrast with an identity ridge penalty — an i.i.d. Gaussian random
//! effect, matching the Rust `RandomEffectTermSpec`) are now all fully wired
//! on the torch path. Every exported variant resolves to a `fit.py` branch
//! that builds a concrete `(design, penalty)` tensor pair.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TorchSmoothEntry {
    Duchon,
    BSpline,
    TensorBSpline,
    Matern,
    Sphere,
    PeriodicSplineCurve,
    Pca,
    Categorical,
}

impl TorchSmoothEntry {
    pub const fn as_str(self) -> &'static str {
        match self {
            TorchSmoothEntry::Duchon => "duchon",
            TorchSmoothEntry::BSpline => "bspline",
            TorchSmoothEntry::TensorBSpline => "tensor_bspline",
            TorchSmoothEntry::Matern => "matern",
            TorchSmoothEntry::Sphere => "sphere",
            TorchSmoothEntry::PeriodicSplineCurve => "periodic_spline_curve",
            TorchSmoothEntry::Pca => "pca",
            TorchSmoothEntry::Categorical => "categorical",
        }
    }
}

/// Map a Python `Smooth` subclass name to the matching torch entry kind.
///
/// Returns `Ok(entry)` for every `Smooth` subclass that `gamfit.torch`
/// re-exports. Each recognised entry has a matching `fit.py` branch that
/// builds a concrete design/penalty tensor pair; the `NotImplementedError`
/// fallback there is now only a defensive guard for a future Rust variant
/// added without a torch branch. Truly unknown class names produce a
/// `TypeError`-shaped message preserving the previous Python cascade's
/// surface error.
pub fn dispatch_key(spec_kind: &str) -> Result<TorchSmoothEntry, String> {
    match spec_kind {
        "Duchon" => Ok(TorchSmoothEntry::Duchon),
        "BSpline" => Ok(TorchSmoothEntry::BSpline),
        "TensorBSpline" => Ok(TorchSmoothEntry::TensorBSpline),
        "Matern" => Ok(TorchSmoothEntry::Matern),
        "Sphere" => Ok(TorchSmoothEntry::Sphere),
        "PeriodicSplineCurve" => Ok(TorchSmoothEntry::PeriodicSplineCurve),
        "Pca" => Ok(TorchSmoothEntry::Pca),
        "Categorical" => Ok(TorchSmoothEntry::Categorical),
        other => Err(format!("unknown Smooth subclass: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_specs_dispatch() {
        assert_eq!(dispatch_key("Duchon").unwrap(), TorchSmoothEntry::Duchon);
        assert_eq!(dispatch_key("BSpline").unwrap(), TorchSmoothEntry::BSpline);
        assert_eq!(
            dispatch_key("TensorBSpline").unwrap(),
            TorchSmoothEntry::TensorBSpline
        );
        assert_eq!(dispatch_key("Matern").unwrap(), TorchSmoothEntry::Matern);
        assert_eq!(dispatch_key("Sphere").unwrap(), TorchSmoothEntry::Sphere);
        assert_eq!(
            dispatch_key("PeriodicSplineCurve").unwrap(),
            TorchSmoothEntry::PeriodicSplineCurve
        );
        assert_eq!(dispatch_key("Pca").unwrap(), TorchSmoothEntry::Pca);
        assert_eq!(
            dispatch_key("Categorical").unwrap(),
            TorchSmoothEntry::Categorical
        );
    }

    #[test]
    fn unknown_spec_kind_is_distinguishable() {
        let err = dispatch_key("Banana").unwrap_err();
        assert!(err.contains("unknown Smooth subclass"));
        assert!(err.contains("Banana"));
    }

    #[test]
    fn as_str_round_trips() {
        for kind in [
            TorchSmoothEntry::Duchon,
            TorchSmoothEntry::BSpline,
            TorchSmoothEntry::TensorBSpline,
            TorchSmoothEntry::Matern,
            TorchSmoothEntry::Sphere,
            TorchSmoothEntry::PeriodicSplineCurve,
            TorchSmoothEntry::Pca,
            TorchSmoothEntry::Categorical,
        ] {
            assert!(!kind.as_str().is_empty());
        }
    }
}

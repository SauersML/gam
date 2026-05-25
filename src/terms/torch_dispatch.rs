//! Dispatch table for the torch fit entry — single source of truth for which
//! Python `Smooth` subclasses the torch.fit autograd glue currently supports.
//!
//! The Python side calls `torch_smooth_dispatch_key(type(smooth).__name__)`
//! to translate the spec class name into a small enumeration. The tensor
//! construction itself stays in Python because the torch autograd VJP must
//! flow back through `points`, `centers`, and `by`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TorchSmoothEntry {
    Duchon,
    BSpline,
    Sphere,
    PeriodicSplineCurve,
    Pca,
}

impl TorchSmoothEntry {
    pub const fn as_str(self) -> &'static str {
        match self {
            TorchSmoothEntry::Duchon => "duchon",
            TorchSmoothEntry::BSpline => "bspline",
            TorchSmoothEntry::Sphere => "sphere",
            TorchSmoothEntry::PeriodicSplineCurve => "periodic_spline_curve",
            TorchSmoothEntry::Pca => "pca",
        }
    }
}

/// Map a Python `Smooth` subclass name to the matching torch entry kind.
///
/// Unsupported but recognised kinds (`TensorBSpline`, `Matern`, `Categorical`)
/// produce the same `NotImplementedError`-shaped message the previous Python
/// cascade raised. Truly unknown class names produce a `TypeError`-shaped
/// message.
pub fn dispatch_key(spec_kind: &str) -> Result<TorchSmoothEntry, String> {
    match spec_kind {
        "Duchon" => Ok(TorchSmoothEntry::Duchon),
        "BSpline" => Ok(TorchSmoothEntry::BSpline),
        "Sphere" => Ok(TorchSmoothEntry::Sphere),
        "PeriodicSplineCurve" => Ok(TorchSmoothEntry::PeriodicSplineCurve),
        "Pca" => Ok(TorchSmoothEntry::Pca),
        "TensorBSpline" | "Matern" | "Categorical" => Err(format!(
            "{spec_kind} not yet wired to gamfit.torch.fit; needs Rust PyO3 \
             binding for the underlying basis + penalty. Currently supported: \
             Duchon (any d for basis; d=1 for penalty), BSpline (d=1), Sphere (S\u{00b2})."
        )),
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
        assert_eq!(dispatch_key("Sphere").unwrap(), TorchSmoothEntry::Sphere);
        assert_eq!(
            dispatch_key("PeriodicSplineCurve").unwrap(),
            TorchSmoothEntry::PeriodicSplineCurve
        );
        assert_eq!(dispatch_key("Pca").unwrap(), TorchSmoothEntry::Pca);
    }

    #[test]
    fn known_unsupported_specs_have_helpful_error() {
        let err = dispatch_key("TensorBSpline").unwrap_err();
        assert!(err.contains("TensorBSpline"));
        assert!(err.contains("Currently supported"));
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
            TorchSmoothEntry::Sphere,
            TorchSmoothEntry::PeriodicSplineCurve,
            TorchSmoothEntry::Pca,
        ] {
            assert!(!kind.as_str().is_empty());
        }
    }
}

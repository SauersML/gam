use gam::terms::torch_dispatch::dispatch_key;
use serde::{Deserialize, Serialize};

#[test]
fn torch_dispatch_maps_every_torch_exported_smooth_class() {
    let exported_torch_smooth_classes = [
        "Duchon",
        "BSpline",
        "TensorBSpline",
        "Matern",
        "Pca",
        "PeriodicSplineCurve",
        "Sphere",
        "Categorical",
    ];

    for class_name in exported_torch_smooth_classes {
        let result = dispatch_key(class_name);
        assert!(
            result.is_ok(),
            "Expected torch dispatch to map exported smooth class '{class_name}' to a backend entry kind, but got error: {result:?}"
        );
    }
}

#[test]
fn unknown_entry_kind_returns_clear_error_without_panicking() {
    let unknown_kind = "TotallyUnknownSmooth";
    let result = dispatch_key(unknown_kind);
    assert!(
        result.is_err(),
        "Expected unknown smooth class '{unknown_kind}' to return an error instead of success"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("unknown Smooth subclass") && err.contains(unknown_kind),
        "Expected unknown smooth error to clearly include both the unknown-kind marker and class name; got: {err}"
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TorchDispatchSpec {
    smooth_class_name: String,
}

#[test]
fn torch_dispatch_spec_round_trip_preserves_dispatch_result() {
    let original = TorchDispatchSpec {
        smooth_class_name: "TensorBSpline".to_string(),
    };

    let encoded =
        serde_json::to_string(&original).expect("Expected TorchDispatchSpec to serialize to JSON");
    let decoded: TorchDispatchSpec = serde_json::from_str(&encoded)
        .expect("Expected TorchDispatchSpec JSON to deserialize back to struct");

    let before = dispatch_key(&original.smooth_class_name);
    let after = dispatch_key(&decoded.smooth_class_name);

    assert!(
        before.is_ok(),
        "Expected original torch-dispatch spec class '{}' to be dispatchable before serialization, but got: {before:?}",
        original.smooth_class_name
    );
    assert_eq!(
        before, after,
        "Expected torch-dispatch result to remain identical after spec serialize/deserialize round-trip"
    );
}

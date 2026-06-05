import pickle

import pytest


def _rust_module():
    try:
        from gamfit._binding import rust_module

        return rust_module()
    except Exception as exc:
        if exc.__class__.__name__ == "RustExtensionUnavailableError":
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize(
    "name",
    [
        "GamError",
        "InvalidInputError",
        "SurvivalMarginalSlopeError",
        "IntegrationError",
    ],
)
def test_rust_exception_classes_are_importably_picklable(name):
    rust = _rust_module()
    cls = getattr(rust, name)

    assert cls.__module__ == "gamfit._rust"
    assert pickle.loads(pickle.dumps(cls)) is cls


@pytest.mark.parametrize(
    "name",
    [
        "InvalidInputError",
        "SurvivalMarginalSlopeError",
        "IntegrationError",
    ],
)
def test_rust_exception_instances_are_importably_picklable(name):
    rust = _rust_module()
    cls = getattr(rust, name)
    original = cls(f"{name} payload")

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is cls
    assert restored.args == original.args

import pickle
from concurrent.futures import ProcessPoolExecutor

import pytest


def _rust_module():
    try:
        from gamfit._binding import rust_module

        return rust_module()
    except Exception as exc:
        if exc.__class__.__name__ == "RustExtensionUnavailableError":
            pytest.skip(str(exc))
        raise


def _rust_exception_instance(name):
    rust = _rust_module()
    cls = getattr(rust, name)
    return cls(f"{name} payload")


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


def test_rust_exception_instance_crosses_process_pool_boundary():
    rust = _rust_module()
    cls = getattr(rust, "IntegrationError")

    with ProcessPoolExecutor(max_workers=1) as pool:
        restored = pool.submit(_rust_exception_instance, "IntegrationError").result(timeout=10)

    assert type(restored) is cls
    assert restored.args == ("IntegrationError payload",)

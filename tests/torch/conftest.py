"""Skip the gamfit.torch test suite when the torch extra is not installed."""

import pytest

pytest.importorskip("torch")

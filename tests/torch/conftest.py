"""Skip the gamfit.torch test suite when the torch extra is not installed."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

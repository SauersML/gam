from ._version import __author__, __author_email__, __version__  # noqa: F401

from .interface import InterfaceMeta
from .decorators import inherit_docs, override, quirk_docs, skip

__all__ = [
    # Interface
    "InterfaceMeta",
    # Standalone decorators
    "inherit_docs",
    "override",
    "skip",
    # Deprecated (will be removed in v2)
    "quirk_docs",
]

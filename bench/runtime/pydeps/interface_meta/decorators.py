import warnings

from .interface import InterfaceMeta
from .utils.inspection import set_skip


def inherit_docs(method=None, mro=True):
    """
    Indicate to `InterfaceMeta` how the wrapped method should be documented.

    Methods need not normally be decorated with this decorator, except in the
    following cases:
    - documentation for quirks should be lifted not from overrides to a
      method, but from some other method (e.g. because subclasses or
      implementations of the interface should override behaviour in a "private"
      method rather than the top-level public method).
    - the method has been nominated by the interface configuration to be
      skipped, in which case decorating with this method will enable
      documentation generation as if it were not.

    Use this decorator as `@inherit_docs([method=...], [mro=...])`.

    Args:
        method (str, None): A method from which documentation for implementation
            specific quirks should be extracted. [Useful when implementations
            of an interface are supposed to change underlying methods rather
            than the public method itself].
        mro (bool): Whether to include documentation from all levels of the
            MRO, starting from the most primitive class that implementated it.
            All higher levels will be considered as "quirks" to the interface's
            definition.

    Returns:
        function: A function wrapper that attaches attributes `_quirks_method` and
        `_quirks_mro` to the method, for interpretation by `InterfaceMeta`.
    """
    return InterfaceMeta.inherit_docs(method=method, mro=mro)


def quirk_docs(method=None, mro=True):
    """
    DEPRECATED: Please use `inherit_docs` instead.
    """
    warnings.warn(
        "The `interface_meta.quirk_docs` decorator has been replaced by `implemented_by` and "
        "will be removed in version 2.0.",
        DeprecationWarning,
    )
    return inherit_docs(method=method, mro=mro)


def override(func=None, force=False, f=None):
    """
    Indicate to `InterfaceMeta` that this method has intentionally overridden an interface method.

    This decorator also allows one to indicate that the method should be
    overridden without warnings even when it does not conform to the API.

    Use this decorator as `@override` or `@override(force=True)`.

    A recommended convention is to use this decorator as the outermost decorator.

    Args:
        func (function, None): The function, if method is decorated by the decorator
            without arguments (e.g. @override), else None.
        force (bool): Whether to force override of method even if the API does
            note match. Note that in this case, documentation is not inherited
            from the MRO.
        f (function, None): Deprecated predecessor of `func`. Maintained for
            backward compatibility until v2.0.

    Returns:
        function: The wrapped function of function wrapper depending on which
            arguments are present.
    """
    if f is not None:
        warnings.warn(
            "The `f` argument to the `interface_meta.override` decorator has been renamed `func`. This "
            "backward compatibility shim will be removed in 2.0.",
            DeprecationWarning,
        )
    return InterfaceMeta.override(func=func or f, force=force)


def skip(func):
    """
    Indicate to `InterfaceMeta` that this method should be skipped.

    Args:
        func (function): The function/member to mark as skipped.

    Returns:
        function: The marked function/member (same instance as that passed in).
    """
    set_skip(func, skip=True)
    return func

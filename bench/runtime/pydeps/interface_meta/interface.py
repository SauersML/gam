from abc import ABCMeta

from .utils.conformance import verify_conformance, verify_not_overridden
from .utils.docs import update_docs
from .utils.inspection import (
    set_explicit_override,
    set_forced_override,
    set_quirk_docs_method,
    set_quirk_docs_mro,
    should_skip,
)


class InterfaceMeta(ABCMeta):
    """
    A metaclass that helps subclasses of a class to conform to its API.

    It also makes sure that documentation that might be useful to a user
    is inherited appropriately, and provides a hook for class to handle
    subclass operations.
    """

    INTERFACE_EXPLICIT_OVERRIDES = True
    INTERFACE_RAISE_ON_VIOLATION = False
    INTERFACE_SKIPPED_NAMES = set()

    def __init__(cls, name, bases, dct):
        ABCMeta.__init__(cls, name, bases, dct)

        # Register interface class for subclasses
        if not hasattr(cls, "__interface__"):
            cls.__interface__ = cls

        # Read configuration
        explicit_overrides = cls.__get_config(
            bases, dct, "INTERFACE_EXPLICIT_OVERRIDES"
        )
        raise_on_violation = cls.__get_config(
            bases, dct, "INTERFACE_RAISE_ON_VIOLATION"
        )
        skipped_names = cls.__get_config(bases, dct, "INTERFACE_SKIPPED_NAMES")

        # Iterate over names in `dct` and check for conformance to interface
        for key, value in dct.items():

            # Skip any key corresponding to Python magic methods
            if key.startswith("__") and key.endswith("__"):
                continue

            # Skip any key in skipped_names
            if key in skipped_names or should_skip(value):  # pragma: no cover
                continue

            # Identify the first instance of this key in the MRO, if it exists, and check conformance
            is_override = False
            for base in cls.__mro__[1:]:
                if base is object:
                    continue
                if key in base.__dict__:
                    is_override = True
                    cls.__verify_conformance(
                        key,
                        name,
                        value,
                        base.__name__,
                        base.__dict__[key],
                        explicit_overrides=explicit_overrides,
                        raise_on_violation=raise_on_violation,
                    )
                    break
                if key in getattr(
                    base, "__annotations__", {}
                ):  # Declared but as yet unspecified attributes
                    is_override = True
                    cls.__verify_conformance(
                        key,
                        name,
                        value,
                        name,
                        None,
                        explicit_overrides=explicit_overrides,
                        raise_on_violation=raise_on_violation,
                    )
                    break

            if not is_override:
                verify_not_overridden(
                    key, name, value, raise_on_violation=raise_on_violation
                )

        # Update documentation
        cls.__update_docs(cls, name, bases, dct)

        # Call subclass registration hook
        cls.__register_implementation__()

    def __register_implementation__(cls):
        pass

    @classmethod
    def __get_config(mcls, bases, dct, key):
        default = getattr(mcls, key, None)
        if bases:
            default = getattr(bases[0], key, default)
        return dct.get(key, default)

    @classmethod
    def __verify_conformance(
        mcls,
        key,
        name,
        value,
        base_name,
        base_value,
        explicit_overrides=True,
        raise_on_violation=False,
    ):
        return verify_conformance(
            key,
            name,
            value,
            base_name,
            base_value,
            explicit_overrides=explicit_overrides,
            raise_on_violation=raise_on_violation,
        )

    @classmethod
    def __update_docs(mcls, cls, name, bases, dct):
        skipped_names = mcls.__get_config(bases, dct, "INTERFACE_SKIPPED_NAMES")
        return update_docs(cls, name, bases, dct, skipped_names=skipped_names)

    @classmethod
    def inherit_docs(mcls, method=None, mro=True):
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

        Use this decorator as `@InterfaceMeta.inherit_docs([method=...], [mro=...])`
        or `@<metaclass instance>.inherit_docs([method=...], [mro=...])`.

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

        def doc_wrapper(f):
            set_quirk_docs_method(f, method)
            set_quirk_docs_mro(f, mro)
            return f

        return doc_wrapper

    @classmethod
    def override(mcls, func=None, force=False):
        """
        Indicate to `InterfaceMeta` that this method has intentionally overridden an interface method.

        This decorator also allows one to indicate that the method should be
        overridden without warnings even when it does not conform to the API.

        Use this decorator as `@InterfaceMeta.override`, `@InterMeta.override(force=True)`,
        `@<metaclass instance>.override`, or `@<metaclass instance>.override(force=True)`.

        A recommended convention is to use this decorator as the outermost decorator.

        Args:
            f (function, None): The function, if method is decorated by the decorator
                without arguments (e.g. @override), else None.
            force (bool): Whether to force override of method even if the API does
                note match. Note that in this case, documentation is not inherited
                from the MRO.

        Returns:
            function: The wrapped function of function wrapper depending on which
                arguments are present.
        """

        def override(f):
            set_explicit_override(f)
            set_forced_override(f, force)
            return f

        if func is not None:
            return override(func)
        return override

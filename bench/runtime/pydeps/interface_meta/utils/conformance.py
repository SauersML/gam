import inspect
import logging
from abc import abstractproperty
from inspect import Parameter

from .inspection import (
    get_functional_signature,
    has_explicit_override,
    has_forced_override,
    is_functional_member,
    is_method,
    should_skip,
)
from .reporting import report_violation


def verify_conformance(
    name,
    clsname,
    member,
    ref_clsname,
    ref_member,
    explicit_overrides=True,
    raise_on_violation=False,
):
    """
    Verify that a member conforms to a nominated interface.

    Args:
        name (str): The name of the member method being checked.
        clsname (str): The name of the class parent of the checked member.
        member (object): The class member to check for conformance against
            ref_member.
        ref_clsname (str): The name of the reference class to be treated as an
            interface.
        ref_member (object): The referece member to be treated as an interface
            definition.
        explicit_overrides (bool): Whether to require explicit overrides.
            (default: True)
        raise_on_violation (bool): Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    if hasattr(
        ref_member, "__objclass__"
    ):  # pragma: no cover; Method is attached to metaclass, so should not be checked.
        return

    if has_forced_override(member) or should_skip(ref_member):
        return

    # Check that type of member has not changed.
    if type(member) is not type(ref_member):
        if type(member) in (abstractproperty, property) and not is_method(ref_member):
            # This should be okay, provided the property is properly crafted.
            pass
        elif is_functional_member(member):
            # This means we are replacing a fixed attribute with a method,
            # or between different types of functional members
            report_violation(
                "`{}.{}` changes the type of `{}.{}` (`{}` instead of `{}`) without using `@override(force=True)` decorator.".format(
                    clsname, name, ref_clsname, name, type(member), type(ref_member)
                ),
                raise_on_violation,
            )
        else:  # Most other type changes should be fine
            pass

    # Check that overrides are present
    if (
        is_functional_member(member)
        or inspect.isdatadescriptor(member)
        or inspect.ismethoddescriptor(member)
    ):
        if explicit_overrides and not has_explicit_override(member):
            report_violation(
                "`{}.{}` overrides interface `{}.{}` without using the `@override` decorator.".format(
                    clsname, name, ref_clsname, name
                ),
                raise_on_violation,
            )

    if is_functional_member(member) and is_functional_member(ref_member):
        verify_signature(
            name,
            clsname,
            member,
            ref_clsname,
            ref_member,
            raise_on_violation=raise_on_violation,
        )


def verify_signature(
    name, clsname, member, ref_clsname, ref_member, raise_on_violation=False
):
    """
    Verify that the signature of a member is compatible with some reference member.

    Args:
        name (str): The name of the member method being checked.
        clsname (str): The name of the class parent of the checked member.
        member (object): The class member to check for conformance against
            ref_member.
        ref_clsname (str): The name of the reference class to be treated as an
            interface.
        ref_member (object): The referece member to be treated as an interface
            definition.
        raise_on_violation (bool): Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    sig = get_functional_signature(member)
    ref_sig = get_functional_signature(ref_member)

    if not check_signatures_compatible(sig, ref_sig):
        message = "Signature `{}.{}{}` does not conform to interface `{}.{}{}`.".format(
            clsname, name, sig, ref_clsname, name, ref_sig
        )
        if raise_on_violation:
            raise RuntimeError(message)
        else:
            logging.warning(message)


def check_signatures_compatible(sig, ref_sig):
    """
    Check whether two signatures are compatible.

    Args:
        sig (Signature): A signature of a member to check for compatibility with
            `ref_sig`.
        ref_sig (Signature): The reference signature.

    Returns:
        bool: `True` if the signatures are compatible, and `False` otherwise.
    """
    params = iter(sig.parameters.values())
    base_params = iter(ref_sig.parameters.values())

    try:
        for bp in base_params:
            cp = next(params)

            while (
                bp.kind is Parameter.VAR_POSITIONAL
                and cp.kind is Parameter.POSITIONAL_OR_KEYWORD
            ):
                cp = next(params)

            while (
                bp.kind is Parameter.VAR_KEYWORD
                and cp.kind is not Parameter.VAR_KEYWORD
            ):
                cp = next(params)

            if not (
                cp.name == bp.name and bp.kind == cp.kind and bp.default == cp.default
            ):
                raise ValueError(bp, cp)

    except (StopIteration, ValueError):
        return False

    for param in params:
        if (
            param.kind is Parameter.POSITIONAL_ONLY
            or param.kind is Parameter.POSITIONAL_OR_KEYWORD
            and param.default == inspect._empty
        ):
            return False

    return True


def verify_not_overridden(name, clsname, member, raise_on_violation=False):
    """
    Verify that a nominated member is *not* an override.

    Args:
        name (str): The name of the member method being checked.
        clsname (str): The name of the class parent of the checked member.
        member (object): The class member to check for conformance against
            ref_member.
        raise_on_violation (bool): Whether any non-conformance should cause an
            exception to be raised. (default: False)
    """
    if has_explicit_override(member):
        report_violation(
            "`{clsname}.{name}` claims to override interface method, but no such method exists.".format(
                clsname=clsname, name=name
            ),
            raise_on_violation=raise_on_violation,
        )

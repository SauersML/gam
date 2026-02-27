import inspect
import textwrap
from collections import OrderedDict

from .inspection import (
    get_class_attr_docs,
    get_functional_docs,
    get_functional_wrapper,
    get_quirk_docs_method,
    get_quirk_docs_mro,
    has_class_attr_docs,
    has_forced_override,
    has_quirk_docs_mro,
    has_updatable_docs,
    set_functional_docs,
)


def update_docs(cls, name, bases, dct, skipped_names=None):
    """
    Update the documentation on class members with information from parents.

    If a method is implementing an interface, its documentation should only need
    to include the quirks associated with the implementation, if any; and all
    other documentation should be inherited. This function performs all of the
    magic required to see that this is the case.

    Args:
        name (str): The name of the class being constructed.
        bases (list<class,type>): The bases of the class being constructed.
        dct (dict): The class dictionary being used to construct the class.
        skipped_names (list<str>): Names for which to skip the documentation
            rewriting.
    """

    mro = inspect.getmro(cls)
    mro = mro[: mro.index(cls.__interface__) + 1]
    skipped_names = skipped_names or set()

    # Handle module-level documentation
    module_docs = [cls.__doc__]
    for klass in mro:
        if has_class_attr_docs(klass):
            module_docs.append(
                [
                    "Attributes:"
                    if klass is cls
                    else "Attributes inherited from {}:".format(klass.__name__),
                    inspect.cleandoc(get_class_attr_docs(klass)),
                ]
            )

    cls.__doc__ = doc_join(*module_docs)

    # Assemble class attribute names avoiding dunder methods
    members = {}
    for klass in reversed(cls.mro()):
        members.update(
            {
                name: member
                for name, member in klass.__dict__.items()
                if not name.startswith("__") and not name.endswith("__")
            }
        )

    # Handle function/method-level documentation
    for name, member in members.items():

        # Skip magic methods
        if name.startswith("__") and name.endswith("__"):
            continue

        # Check if there is anything to do
        if not has_updatable_docs(member):
            continue

        quirks_method = get_quirk_docs_method(member)
        quirks_mro = get_quirk_docs_mro(member)
        has_quirks_mro = has_quirk_docs_mro(member)

        if (
            inspect.isabstract(member)
            or has_forced_override(member)
            or name in skipped_names
            and not (has_quirks_mro or quirks_method)
            or name not in cls.__dict__
            and quirks_method is None
        ):
            continue

        # Extract documentation from this member and the quirks member
        method_docs = OrderedDict()
        last_docs = None
        for i, klass in enumerate(reversed(mro) if quirks_mro else mro[:1]):
            klass_member = klass.__dict__.get(name, None)
            if klass_member is not None:
                member_docs = get_functional_docs(klass_member)
                if (i == 0 or member_docs) and member_docs != last_docs:
                    last_docs = method_docs[klass.__name__] = member_docs
                if not get_quirk_docs_mro(klass_member):
                    break

        if quirks_method is not None and quirks_method in members:
            quirk_member = members.get(quirks_method)
            quirk_member_docs = get_functional_docs(quirk_member)
            if quirk_member_docs:
                if cls.__name__ in method_docs:
                    method_docs[cls.__name__] = (
                        inspect.cleandoc(method_docs[cls.__name__])
                        + "\n\n"
                        + inspect.cleandoc(quirk_member_docs)
                    )
                else:
                    method_docs[cls.__name__] = quirk_member_docs

        if method_docs:

            if name not in cls.__dict__:
                # Overide method object with new object so we don't modify
                # underlying method that may be shared by multiple classes.
                member = get_functional_wrapper(member)

            set_functional_docs(
                member,
                doc_join(
                    *[
                        docs if i == 0 else [source + " Quirks:", docs]
                        for i, (source, docs) in enumerate(method_docs.items())
                    ]
                ),
            )

            if name not in cls.__dict__:
                setattr(cls, name, member)


def doc_join(*docs):
    """
    Stitch multiple pieces of documentation into one docstring.

    Args:
        *docs (tuple<str, list<str>, tuple<str>>): A sequence of strings or
            length-2 sequences of strings to stitch together into a docstring.
            If a length-2 sequence is provided, then the first string is treated
            as a section header, and the rest of the string is indented
            beneath it.

    Returns:
        str: The stitched together docstring.
    """
    out = []
    for doc in docs:
        if doc in (None, ""):
            continue
        elif isinstance(doc, str):
            out.append(textwrap.dedent(doc).strip("\n"))
        elif isinstance(doc, (list, tuple)):
            if len(doc) < 2:
                continue
            d = doc_join(*doc[1:])
            if d:
                if not out:
                    out.append("\n")
                out.append(
                    "{header}\n{body}".format(
                        header=doc[0].strip(),
                        body="    "
                        + d.replace(
                            "\n", "\n    "
                        ),  # textwrap.indent not available in python2
                    )
                )
        else:
            raise ValueError("Unrecognised doc format: {}".format(type(doc)))
    return "\n\n".join(out) or None

#!/usr/bin/env python3
"""Generate ``gamfit/_rust.pyi`` from the PyO3 source of ``crates/gam-pyffi``.

The compiled ``gamfit._rust`` extension is the only untyped surface of the
package. Hand-maintaining its stub would drift the moment a ``#[pyfunction]``
gains an argument, so the stub is derived from the Rust declarations:

* the registrations (``add_function(wrap_pyfunction!(f, ..))``,
  ``add_class::<T>()`` and ``module.add("Name", ..)``) decide which names the
  module exports;
* ``#[pyfunction]`` / ``#[pymethods]`` items and their ``#[pyo3(signature =
  ...)]`` attributes decide parameter names, kinds and defaults;
* the Rust argument and return types decide the Python types, through the
  conversion rules PyO3 and rust-numpy apply at the boundary.

A Rust type the rules do not cover is an error, never a silent ``Any``: the
generator refuses to emit a stub it cannot type. ``--check`` compares the
committed stub with a fresh generation and fails on any difference; the test
suite runs it, together with ``mypy.stubtest`` against the compiled module.

Usage::

    python scripts/gen_rust_stub.py            # rewrite gamfit/_rust.pyi
    python scripts/gen_rust_stub.py --check    # exit 1 if the stub is stale
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "crates" / "gam-pyffi" / "src"
STUB = ROOT / "gamfit" / "_rust.pyi"
PROTOCOL = ROOT / "gamfit" / "_rust_module.pyi"


# --------------------------------------------------------------------------
# Lexer


@dataclasses.dataclass(frozen=True)
class Token:
    kind: str  # "ident", "lifetime", "str", "num", "char", "punct"
    text: str
    origin: str


_IDENT = re.compile(r"r#[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*")
_LIFETIME = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*(?!')")
_CHAR = re.compile(r"b?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'])'")
_NUMBER = re.compile(r"[0-9][0-9A-Za-z_]*(?:\.[0-9][0-9A-Za-z_]*)?(?:[eE][+-]?[0-9_]+)?")
_RAW_STRING = re.compile(r'(?:b|c)?r(#*)"')
_STRING = re.compile(r'(?:b|c)?"((?:\\.|[^"\\])*)"', re.DOTALL)
_PUNCT = ("::", "->", "=>", "..=", "...", "..", "==", "!=", "<=", ">=", "&&", "||")


def lex(source: str, origin: str) -> list[Token]:
    """Tokenize Rust source, keeping string literal values."""
    out: list[Token] = []
    position = 0
    length = len(source)
    while position < length:
        char = source[position]
        if char.isspace():
            position += 1
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position)
            position = length if newline < 0 else newline + 1
            continue
        if source.startswith("/*", position):
            depth = 1
            position += 2
            while depth:
                opening = source.find("/*", position)
                closing = source.find("*/", position)
                if closing < 0:
                    raise ValueError(f"{origin}: unterminated block comment")
                if 0 <= opening < closing:
                    depth += 1
                    position = opening + 2
                else:
                    depth -= 1
                    position = closing + 2
            continue
        raw = _RAW_STRING.match(source, position)
        if raw:
            marker = '"' + raw.group(1)
            end = source.find(marker, raw.end())
            if end < 0:
                raise ValueError(f"{origin}: unterminated raw string")
            out.append(Token("str", source[raw.end() : end], origin))
            position = end + len(marker)
            continue
        string = _STRING.match(source, position)
        if string:
            out.append(Token("str", _unescape(string.group(1)), origin))
            position = string.end()
            continue
        character = _CHAR.match(source, position)
        if character:
            out.append(Token("char", character.group(), origin))
            position = character.end()
            continue
        lifetime = _LIFETIME.match(source, position)
        if lifetime:
            out.append(Token("lifetime", lifetime.group(), origin))
            position = lifetime.end()
            continue
        ident = _IDENT.match(source, position)
        if ident:
            text = ident.group()
            out.append(Token("ident", text[2:] if text.startswith("r#") else text, origin))
            position = ident.end()
            continue
        number = _NUMBER.match(source, position)
        if number:
            out.append(Token("num", number.group(), origin))
            position = number.end()
            continue
        for punct in _PUNCT:
            if source.startswith(punct, position):
                out.append(Token("punct", punct, origin))
                position += len(punct)
                break
        else:
            out.append(Token("punct", char, origin))
            position += 1
    return out


def _unescape(body: str) -> str:
    return re.sub(r"\\\n\s*", "", body).replace('\\"', '"').replace("\\\\", "\\")


_OPEN = {"(": ")", "[": "]", "{": "}"}


def matching(tokens: Sequence[Token], start: int) -> int:
    """Index of the bracket closing ``tokens[start]`` (``(``, ``[`` or ``{``)."""
    stack = [_OPEN[tokens[start].text]]
    index = start + 1
    while stack:
        token = tokens[index]
        if token.kind == "punct":
            if token.text in _OPEN:
                stack.append(_OPEN[token.text])
            elif token.text == stack[-1]:
                stack.pop()
            elif token.text in (")", "]", "}"):
                raise ValueError(f"{token.origin}: unbalanced {token.text!r}")
        index += 1
    return index - 1


def split_top_level(tokens: Sequence[Token]) -> list[list[Token]]:
    """Split on commas outside every bracket and every generic ``<...>``."""
    parts: list[list[Token]] = [[]]
    depth = 0
    for token in tokens:
        if token.kind == "punct":
            if token.text in ("(", "[", "{", "<"):
                depth += 1
            elif token.text in (")", "]", "}", ">"):
                depth -= 1
            elif token.text == "->":
                pass
            elif token.text == "," and depth == 0:
                parts.append([])
                continue
        parts[-1].append(token)
    return [part for part in parts if part]


def text(tokens: Sequence[Token]) -> str:
    return " ".join(token.text for token in tokens)


# --------------------------------------------------------------------------
# Rust items


@dataclasses.dataclass
class Attribute:
    name: str
    body: list[Token]


@dataclasses.dataclass
class RustArg:
    name: str
    ty: list[Token]


@dataclasses.dataclass
class RustFn:
    name: str
    attrs: list[Attribute]
    args: list[RustArg]
    ret: list[Token]
    receiver: str | None  # None, "self", "cls"
    origin: str


@dataclasses.dataclass
class RustClass:
    rust_name: str
    python_name: str
    fields: list[tuple[str, list[Token], bool]]  # (python name, type, settable)
    origin: str
    is_enum: bool
    variants: list[str]
    methods: list[RustFn] = dataclasses.field(default_factory=list)
    class_attrs: list[tuple[str, list[Token]]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Registration:
    functions: list[str] = dataclasses.field(default_factory=list)
    classes: list[str] = dataclasses.field(default_factory=list)
    values: list[tuple[str, list[Token]]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Exception_:
    name: str
    base: str


@dataclasses.dataclass
class Crate:
    functions: dict[str, RustFn] = dataclasses.field(default_factory=dict)
    classes: dict[str, RustClass] = dataclasses.field(default_factory=dict)
    exceptions: dict[str, Exception_] = dataclasses.field(default_factory=dict)
    registration: Registration = dataclasses.field(default_factory=Registration)
    pending_methods: list[tuple[str, list[RustFn], list[tuple[str, list[Token]]]]] = (
        dataclasses.field(default_factory=list)
    )
    constants: dict[str, list[Token]] = dataclasses.field(default_factory=dict)
    # `#[derive(FromPyObject)]` enums: Rust name -> the field types of each
    # tuple variant, tried in order by PyO3's extraction.
    extractions: dict[str, list[list[list[Token]]]] = dataclasses.field(default_factory=dict)


def parse_attribute(tokens: Sequence[Token], index: int) -> tuple[Attribute, int]:
    """Parse ``#[...]`` starting at ``tokens[index] == '#'``."""
    close = matching(tokens, index + 1)
    body = list(tokens[index + 2 : close])
    name = "::".join(t.text for t in body[: _path_end(body)] if t.text != "::")
    return Attribute(name.split("::")[-1] if name else "", body), close + 1


def _path_end(body: Sequence[Token]) -> int:
    index = 0
    while index < len(body) and (body[index].kind == "ident" or body[index].text == "::"):
        index += 1
    return index


def attribute_args(attribute: Attribute) -> list[list[Token]]:
    """The comma-separated arguments of ``#[name(...)]``."""
    body = attribute.body
    start = _path_end(body)
    if start >= len(body) or body[start].text != "(":
        return []
    close = matching(body, start)
    return split_top_level(body[start + 1 : close])


def keyed(attribute: Attribute, key: str) -> list[Token] | None:
    for arg in attribute_args(attribute):
        if arg and arg[0].text == key and len(arg) > 1 and arg[1].text == "=":
            return arg[2:]
    return None


def flag(attribute: Attribute, key: str) -> bool:
    return any(len(arg) == 1 and arg[0].text == key for arg in attribute_args(attribute))


def pyo3_option(attrs: Sequence[Attribute], key: str) -> list[Token] | None:
    for attribute in attrs:
        if attribute.name == "pyo3":
            found = keyed(attribute, key)
            if found is not None:
                return found
    return None


def has_attr(attrs: Sequence[Attribute], name: str) -> bool:
    return any(attribute.name == name for attribute in attrs)


def find_attr(attrs: Sequence[Attribute], name: str) -> Attribute | None:
    for attribute in attrs:
        if attribute.name == name:
            return attribute
    return None


_QUALIFIERS = {"pub", "const", "async", "unsafe", "extern", "default"}


def skip_visibility(tokens: Sequence[Token], index: int) -> int:
    while index < len(tokens) and tokens[index].text in _QUALIFIERS:
        if tokens[index].text == "pub" and tokens[index + 1].text == "(":
            index = matching(tokens, index + 1) + 1
        else:
            index += 1
    return index


def skip_generics(tokens: Sequence[Token], index: int) -> int:
    if index >= len(tokens) or tokens[index].text != "<":
        return index
    depth = 0
    while True:
        token = tokens[index]
        if token.text == "<":
            depth += 1
        elif token.text == ">":
            depth -= 1
            if depth == 0:
                return index + 1
        elif token.text == "->":
            pass
        index += 1


def _is_const_item(tokens: Sequence[Token], name: int) -> bool:
    """Whether ``tokens[name]`` names a ``const NAME: T = ..`` item.

    ``skip_visibility`` consumes the ``const`` keyword (it also qualifies
    ``const fn``), so the item is recognised from the keyword before the name.
    """
    return (
        0 < name < len(tokens) - 1
        and tokens[name - 1].text == "const"
        and tokens[name].kind == "ident"
        and tokens[name + 1].text == ":"
    )


def parse_fn(tokens: Sequence[Token], index: int, attrs: list[Attribute]) -> tuple[RustFn, int]:
    """Parse ``fn name<..>(args) -> ret`` with ``tokens[index] == 'fn'``."""
    name = tokens[index + 1].text
    index = skip_generics(tokens, index + 2)
    close = matching(tokens, index)
    params = split_top_level(tokens[index + 1 : close])
    index = close + 1
    ret: list[Token] = []
    if tokens[index].text == "->":
        index += 1
        while tokens[index].text not in ("{", ";", "where"):
            ret.append(tokens[index])
            index += 1
    while tokens[index].text not in ("{", ";"):
        index += 1
    end = matching(tokens, index) + 1 if tokens[index].text == "{" else index + 1
    receiver: str | None = None
    args: list[RustArg] = []
    for position, param in enumerate(params):
        words = [t.text for t in param]
        if "self" in words[:3] and ":" not in words[: words.index("self") + 1]:
            receiver = "self"
            continue
        colon = next(i for i, t in enumerate(param) if t.text == ":")
        pattern = [t for t in param[:colon] if t.text != "mut"]
        ty = param[colon + 1 :]
        arg_name = pattern[-1].text
        if position == 0 and arg_name in ("slf", "self_") and _is_self_ref(ty):
            receiver = "self"
            continue
        args.append(RustArg(arg_name, ty))
    return RustFn(name, attrs, args, ret, receiver, tokens[index].origin), end


def _is_self_ref(ty: Sequence[Token]) -> bool:
    words = [t.text for t in ty]
    return "Self" in words or any(w in ("PyRef", "PyRefMut") for w in words)


def parse_struct(
    tokens: Sequence[Token], index: int, attrs: list[Attribute], crate: Crate
) -> int:
    """Parse a ``#[pyclass]`` struct or enum with ``tokens[index]`` its keyword."""
    is_enum = tokens[index].text == "enum"
    rust_name = tokens[index + 1].text
    pyclass = find_attr(attrs, "pyclass")
    assert pyclass is not None
    renamed = keyed(pyclass, "name")
    python_name = renamed[0].text if renamed else rust_name
    get_all = flag(pyclass, "get_all")
    set_all = flag(pyclass, "set_all")
    index = skip_generics(tokens, index + 2)
    while tokens[index].text not in ("{", "(", ";"):
        index += 1
    fields: list[tuple[str, list[Token], bool]] = []
    variants: list[str] = []
    end = index + 1
    if tokens[index].text in ("{", "("):
        close = matching(tokens, index)
        end = close + 1
        if tokens[index].text == "(":
            end += 1  # trailing ';'
        for part in split_top_level(tokens[index + 1 : close]):
            position = 0
            field_attrs: list[Attribute] = []
            while position < len(part) and part[position].text == "#":
                attribute, position = parse_attribute(part, position)
                field_attrs.append(attribute)
            position = skip_visibility(part, position)
            if is_enum:
                variants.append(part[position].text)
                continue
            if position + 1 >= len(part) or part[position + 1].text != ":":
                continue
            field_name = part[position].text
            field_ty = part[position + 2 :]
            getter = get_all
            setter = set_all
            python_field = field_name
            for attribute in field_attrs:
                if attribute.name == "pyo3":
                    getter = getter or flag(attribute, "get")
                    setter = setter or flag(attribute, "set")
                    rename = keyed(attribute, "name")
                    if rename:
                        python_field = rename[0].text
            if getter or setter:
                fields.append((python_field, field_ty, setter))
    crate.classes[rust_name] = RustClass(
        rust_name, python_name, fields, tokens[index].origin, is_enum, variants
    )
    return end


def derives_from_py_object(attrs: Sequence[Attribute]) -> bool:
    return any(
        attribute.name == "derive"
        and any(arg and arg[-1].text == "FromPyObject" for arg in attribute_args(attribute))
        for attribute in attrs
    )


def parse_extraction_enum(tokens: Sequence[Token], index: int, crate: Crate) -> int:
    """Parse a ``#[derive(FromPyObject)] enum`` with ``tokens[index] == 'enum'``.

    PyO3 extracts such an enum by trying each variant in order: a one-field
    tuple variant accepts what its field accepts, a wider one a tuple of its
    fields. Struct variants extract by attribute name, which no annotation
    here expresses, so they are refused.
    """
    rust_name = tokens[index + 1].text
    index = skip_generics(tokens, index + 2)
    close = matching(tokens, index)
    variants: list[list[list[Token]]] = []
    for part in split_top_level(tokens[index + 1 : close]):
        position = 0
        while position < len(part) and part[position].text == "#":
            _, position = parse_attribute(part, position)
        if position + 1 >= len(part) or part[position + 1].text != "(":
            raise ValueError(
                f"{part[position].origin}: FromPyObject enum {rust_name} needs tuple variants"
            )
        fields = matching(part, position + 1)
        variants.append(split_top_level(part[position + 2 : fields]))
    crate.extractions[rust_name] = variants
    return close + 1


def parse_impl(tokens: Sequence[Token], index: int, crate: Crate) -> int:
    """Parse a ``#[pymethods] impl Type { ... }`` block."""
    index = skip_generics(tokens, index + 1)
    start = index
    while tokens[index].text != "{":
        index += 1
    type_tokens = [t for t in tokens[start:index] if t.kind == "ident"]
    rust_name = type_tokens[-1].text
    close = matching(tokens, index)
    body = tokens[index + 1 : close]
    methods: list[RustFn] = []
    class_attrs: list[tuple[str, list[Token]]] = []
    position = 0
    attrs: list[Attribute] = []
    while position < len(body):
        token = body[position]
        if token.text == "#":
            attribute, position = parse_attribute(body, position)
            attrs.append(attribute)
            continue
        position = skip_visibility(body, position)
        if body[position].text == "fn":
            method, position = parse_fn(body, position, attrs)
            methods.append(method)
            attrs = []
            continue
        if _is_const_item(body, position) and has_attr(attrs, "classattr"):
            equals = next(i for i in range(position, len(body)) if body[i].text == "=")
            class_attrs.append((body[position].text, list(body[position + 2 : equals])))
        while body[position].text != ";":
            position += 1
        attrs = []
        position += 1
    crate.pending_methods.append((rust_name, methods, class_attrs))
    return close + 1


def scan_registrations(tokens: Sequence[Token], crate: Crate) -> None:
    """Collect module registrations and ``create_exception!`` declarations."""
    reg = crate.registration
    for index, token in enumerate(tokens):
        if token.text == "wrap_pyfunction" and tokens[index + 1].text == "!":
            open_index = index + 2
            close = matching(tokens, open_index)
            path = split_top_level(tokens[open_index + 1 : close])[0]
            reg.functions.append(path[-1].text)
        elif token.text == "add_class" and tokens[index + 1].text == "::":
            end = skip_generics(tokens, index + 2)
            path = [t for t in tokens[index + 3 : end - 1] if t.kind == "ident"]
            reg.classes.append(path[-1].text)
        elif (
            token.text == "add"
            and index > 0
            and tokens[index - 1].text == "."
            and tokens[index + 1].text == "("
            and tokens[index + 2].kind == "str"
        ):
            close = matching(tokens, index + 1)
            args = split_top_level(tokens[index + 2 : close])
            if len(args) == 2:
                reg.values.append((args[0][0].text, args[1]))
        elif token.text == "create_exception" and tokens[index + 1].text == "!":
            close = matching(tokens, index + 2)
            args = split_top_level(tokens[index + 3 : close])
            base = [t for t in args[2] if t.kind == "ident"][-1].text
            crate.exceptions[args[1][0].text] = Exception_(args[1][0].text, base)


def parse_file(path: Path, crate: Crate) -> None:
    origin = str(path.relative_to(ROOT))
    tokens = lex(path.read_text(), origin)
    scan_registrations(tokens, crate)
    index = 0
    attrs: list[Attribute] = []
    while index < len(tokens):
        token = tokens[index]
        if token.text == "#" and index + 1 < len(tokens):
            if tokens[index + 1].text == "!":
                index = matching(tokens, index + 2) + 1
                continue
            if tokens[index + 1].text == "[":
                attribute, index = parse_attribute(tokens, index)
                attrs.append(attribute)
                continue
        position = skip_visibility(tokens, index)
        keyword = tokens[position].text if position < len(tokens) else ""
        if keyword == "fn" and has_attr(attrs, "pyfunction"):
            function, index = parse_fn(tokens, position, attrs)
            crate.functions[function.name] = function
            attrs = []
            continue
        if keyword in ("struct", "enum") and has_attr(attrs, "pyclass"):
            index = parse_struct(tokens, position, attrs, crate)
            attrs = []
            continue
        if keyword == "enum" and derives_from_py_object(attrs):
            index = parse_extraction_enum(tokens, position, crate)
            attrs = []
            continue
        if keyword == "impl" and has_attr(attrs, "pymethods"):
            index = parse_impl(tokens, position, crate)
            attrs = []
            continue
        if _is_const_item(tokens, position):
            end = position + 2
            while tokens[end].text != "=":
                end += 1
            crate.constants[keyword] = list(tokens[position + 2 : end])
        attrs = []
        index += 1


def parse_crate(source: Path) -> Crate:
    crate = Crate()
    for path in sorted(source.rglob("*.rs")):
        parse_file(path, crate)
    for rust_name, methods, class_attrs in crate.pending_methods:
        if rust_name not in crate.classes:
            raise ValueError(f"#[pymethods] for unknown #[pyclass] {rust_name}")
        crate.classes[rust_name].methods.extend(methods)
        crate.classes[rust_name].class_attrs.extend(class_attrs)
    return crate


# --------------------------------------------------------------------------
# Rust types


@dataclasses.dataclass(frozen=True)
class Ty:
    """A Rust type with lifetimes and references erased.

    ``name`` is the last path segment (``numpy::PyReadonlyArray2`` becomes
    ``PyReadonlyArray2``); tuples use ``"()"`` and slices ``"[]"``.
    """

    name: str
    args: tuple[Ty, ...] = ()
    origin: str = ""


def parse_type(tokens: Sequence[Token]) -> Ty:
    ty, rest = _parse_type(list(tokens), 0)
    if rest != len(tokens):
        raise ValueError(f"{tokens[0].origin}: cannot parse type {text(tokens)!r}")
    return ty


def _parse_type(tokens: list[Token], index: int) -> tuple[Ty, int]:
    while tokens[index].text in ("&", "mut", "dyn", "impl") or tokens[index].kind == "lifetime":
        index += 1
    origin = tokens[index].origin
    if tokens[index].text in ("(", "["):
        close = matching(tokens, index)
        opener = tokens[index].text
        parts = split_top_level(tokens[index + 1 : close])
        if opener == "[":
            # `[T]` or `[T; N]`: the element type precedes any `;`.
            element = parts[0]
            semicolon = next((i for i, t in enumerate(element) if t.text == ";"), len(element))
            return Ty("[]", (parse_type(element[:semicolon]),), origin), close + 1
        return Ty("()", tuple(parse_type(part) for part in parts), origin), close + 1
    name = tokens[index].text
    index += 1
    while index < len(tokens) and tokens[index].text == "::":
        name = tokens[index + 1].text
        index += 2
    args: list[Ty] = []
    if index < len(tokens) and tokens[index].text == "<":
        close = skip_generics(tokens, index) - 1
        for part in split_top_level(tokens[index + 1 : close]):
            if all(t.kind == "lifetime" for t in part):
                continue
            args.append(parse_type(part))
        index = close + 1
    return Ty(name, tuple(args), origin), index


# --------------------------------------------------------------------------
# Rust type -> Python annotation
#
# The rules mirror the conversions PyO3 and rust-numpy perform at the
# boundary. An argument is typed by what extraction accepts (a Rust `Vec<T>`
# extracts from any sequence, so it takes `Sequence[T]`); a return value by
# what conversion produces (the same `Vec<T>` becomes a `list[T]`).

_INT = {"i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize"}
_FLOAT = {"f32", "f64"}
_STR = {"String", "str", "char", "PyString", "Cow"}
_NUMPY_DTYPE = {
    "f64": "np.float64",
    "f32": "np.float32",
    "i64": "np.int64",
    "i32": "np.int32",
    "i16": "np.int16",
    "i8": "np.int8",
    "u64": "np.uint64",
    "u32": "np.uint32",
    "u16": "np.uint16",
    "u8": "np.uint8",
    "usize": "np.uintp",
    "isize": "np.intp",
    "bool": "np.bool_",
}
_NUMPY_ARRAY = re.compile(r"Py(?:Readonly|Readwrite)?Array(?:[0-6]|Dyn)?")
# Wrappers that change ownership or borrowing but not the Python type.
_TRANSPARENT = {"Bound", "Py", "PyRef", "PyRefMut", "Borrowed", "Box", "Arc", "Rc"}
_MAPS = {"HashMap", "BTreeMap", "IndexMap"}
_SETS = {"HashSet", "BTreeSet"}
# Python-native object types, as annotations for (argument, return value).
_NATIVE = {
    "PyAny": ("object", "Any"),
    "PyObject": ("object", "Any"),
    "PyDict": ("dict[Any, Any]", "dict[Any, Any]"),
    "PyList": ("list[Any]", "list[Any]"),
    "PyTuple": ("tuple[Any, ...]", "tuple[Any, ...]"),
    "PyBytes": ("bytes", "bytes"),
    "PyFloat": ("float", "float"),
    "PyInt": ("int", "int"),
    "PyBool": ("bool", "bool"),
    "PyType": ("type", "type"),
}


class TypeMapper:
    """Rust type -> Python annotation; ``qualifier`` prefixes the module's own classes."""

    def __init__(self, crate: Crate, qualifier: str = "") -> None:
        self.classes = {rust: qualifier + cls.python_name for rust, cls in crate.classes.items()}
        self.extractions = crate.extractions

    def annotate(self, ty: Ty, *, argument: bool, owner: str | None) -> str:
        name, args = ty.name, ty.args
        if name == "Self":
            if owner is None:
                raise ValueError(f"{ty.origin}: `Self` outside an impl")
            return owner
        if name in self.classes:
            return self.classes[name]
        if name in self.extractions:
            if not argument:
                raise ValueError(f"{ty.origin}: FromPyObject enum {name} is never returned")
            return " | ".join(
                self._extracted_variant(fields, owner) for fields in self.extractions[name]
            )
        if name in _TRANSPARENT:
            (wrapped,) = args
            return self.annotate(wrapped, argument=argument, owner=owner)
        if name in ("PyResult", "Result"):
            return self.annotate(args[0], argument=argument, owner=owner)
        if name == "Option":
            return f"{self.annotate(args[0], argument=argument, owner=owner)} | None"
        if name in _INT:
            return "int"
        if name in _FLOAT:
            return "float"
        if name == "bool":
            return "bool"
        if name in _STR:
            return "str"
        if name in ("PathBuf", "Path"):
            return "str | os.PathLike[str]" if argument else "pathlib.Path"
        if name in _NATIVE:
            return _NATIVE[name][0 if argument else 1]
        if _NUMPY_ARRAY.fullmatch(name):
            dtype = args[0].name
            if dtype not in _NUMPY_DTYPE:
                raise ValueError(f"{ty.origin}: no numpy dtype for array element {dtype}")
            return f"NDArray[{_NUMPY_DTYPE[dtype]}]"
        if name in ("Vec", "[]"):
            (element,) = args
            if element.name == "u8":
                # PyO3 converts byte vectors and slices to `bytes`; extraction
                # into `Vec<u8>` still walks any integer sequence.
                return "Sequence[int]" if argument and name == "Vec" else "bytes"
            inner = self.annotate(element, argument=argument, owner=owner)
            return f"Sequence[{inner}]" if argument else f"list[{inner}]"
        if name in _MAPS:
            key, value = (self.annotate(a, argument=argument, owner=owner) for a in args[:2])
            return f"dict[{key}, {value}]"
        if name in _SETS:
            return f"set[{self.annotate(args[0], argument=argument, owner=owner)}]"
        if name == "()":
            if not args:
                return "None"
            return "tuple[" + ", ".join(
                self.annotate(a, argument=argument, owner=owner) for a in args
            ) + "]"
        raise ValueError(f"{ty.origin}: no Python type for Rust type `{name}`")

    def _extracted_variant(self, fields: Sequence[Sequence[Token]], owner: str | None) -> str:
        types = [self.annotate(parse_type(field), argument=True, owner=owner) for field in fields]
        return types[0] if len(types) == 1 else "tuple[" + ", ".join(types) + "]"


# --------------------------------------------------------------------------
# Python signatures


@dataclasses.dataclass
class Param:
    name: str
    annotation: str
    kind: str  # "positional", "normal", "keyword", "varargs", "kwargs"
    default: bool


def _is_python_token(ty: Ty) -> bool:
    return ty.name == "Python"


def signature_attribute(function: RustFn) -> tuple[list[list[Token]], str | None] | None:
    """The items of ``signature = (..)`` and its ``-> "T"`` override, if any."""
    for attribute in function.attrs:
        if attribute.name not in ("pyo3", "pyfunction"):
            continue
        value = keyed(attribute, "signature")
        if value is None:
            continue
        close = matching(value, 0)
        returns: str | None = None
        if close + 2 < len(value) + 1 and close + 1 < len(value) and value[close + 1].text == "->":
            returns = value[close + 2].text
        return split_top_level(value[1:close]), returns
    return None


def python_params(
    function: RustFn, mapper: TypeMapper, owner: str | None, *, drop_first: bool
) -> tuple[list[Param], str | None]:
    """Parameters as Python sees them, plus a return-annotation override."""
    rust_args = [arg for arg in function.args if not _is_python_token(parse_type(arg.ty))]
    if drop_first:
        rust_args = rust_args[1:]
    types = {
        arg.name: mapper.annotate(parse_type(arg.ty), argument=True, owner=owner)
        for arg in rust_args
    }
    declared = signature_attribute(function)
    if declared is None:
        return [Param(arg.name, types[arg.name], "normal", False) for arg in rust_args], None
    items, returns = declared
    params: list[Param] = []
    kind = "normal"
    for item in items:
        words = [t.text for t in item]
        if words == ["/"]:
            for param in params:
                param.kind = "positional"
            continue
        if words == ["*"]:
            kind = "keyword"
            continue
        if words[0] == "*":
            params.append(Param(words[1], "Any", "varargs", False))
            kind = "keyword"
            continue
        if words[0] == "**":
            params.append(Param(words[1], "Any", "kwargs", False))
            continue
        name = words[0]
        annotation = types.get(name)
        if annotation is None:
            raise ValueError(f"{function.origin}: signature names unknown argument {name!r}")
        if len(item) > 2 and words[1] == ":":
            annotation = item[2].text
        params.append(Param(name, annotation, kind, "=" in words))
    declared_names = {param.name for param in params}
    missing = [arg.name for arg in rust_args if arg.name not in declared_names]
    if missing:
        raise ValueError(f"{function.origin}: {function.name} signature omits {missing}")
    return params, returns


def render_params(params: Sequence[Param], receiver: str | None) -> str:
    rendered: list[str] = [receiver] if receiver else []
    positional_only = any(param.kind == "positional" for param in params)
    star_written = False
    for param in params:
        if param.kind == "varargs":
            rendered.append(f"*{param.name}: {param.annotation}")
            star_written = True
            continue
        if param.kind == "kwargs":
            rendered.append(f"**{param.name}: {param.annotation}")
            continue
        if param.kind == "keyword" and not star_written:
            rendered.append("*")
            star_written = True
        default = " = ..." if param.default else ""
        rendered.append(f"{param.name}: {param.annotation}{default}")
        if positional_only and param.kind == "positional" and (
            param is [p for p in params if p.kind == "positional"][-1]
        ):
            rendered.append("/")
    return ", ".join(rendered)


# --------------------------------------------------------------------------
# Stub emission

HEADER = '''\
# Generated by scripts/gen_rust_stub.py from crates/gam-pyffi/src -- do not edit.
# Regenerate with `python scripts/gen_rust_stub.py`; the test suite fails on drift.

import os
import pathlib
from collections.abc import Sequence
from typing import Any, ClassVar, final

import numpy as np
from numpy.typing import NDArray
'''

PROTOCOL_HEADER = '''\
# Generated by scripts/gen_rust_stub.py from crates/gam-pyffi/src -- do not edit.
# Regenerate with `python scripts/gen_rust_stub.py`; the test suite fails on drift.
"""The static type of ``gamfit._binding.rust_module()``.

A function cannot be annotated as returning one particular module, so the
attributes of ``gamfit._rust`` are restated structurally, from the same Rust
declarations as ``gamfit/_rust.pyi``; every Python -> Rust call through
``rust_module()`` is checked against the Rust signatures.
"""

import os
import pathlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from . import _rust
'''

# Receiver-less protocol methods whose argument types PyO3 widens.
_COMPARISONS = {"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"}


def _is_slot(name: str) -> bool:
    """PyO3 implements a magic method as a type slot, whose arguments are
    positional-only; ``__call__`` is the one it binds as an ordinary method."""
    return name.startswith("__") and name.endswith("__") and name != "__call__"


def _return(function: RustFn, mapper: TypeMapper, owner: str | None) -> str:
    if not function.ret:
        return "None"
    return mapper.annotate(parse_type(function.ret), argument=False, owner=owner)


def emit_function(function: RustFn, mapper: TypeMapper, indent: str = "") -> list[str]:
    params, returns = python_params(function, mapper, None, drop_first=False)
    ret = returns or _return(function, mapper, None)
    return [f"{indent}def {function.name}({render_params(params, None)}) -> {ret}: ..."]


def _accessor_name(method: RustFn, attribute: str, prefix: str) -> str:
    found = find_attr(method.attrs, attribute)
    assert found is not None
    args = attribute_args(found)
    if args:
        return args[0][0].text
    return method.name[len(prefix) :] if method.name.startswith(prefix) else method.name


def emit_class(cls: RustClass, mapper: TypeMapper) -> list[str]:
    owner = cls.python_name
    lines = ["@final", f"class {owner}:"]
    body: list[tuple[str, list[str]]] = []
    for name, ty in cls.class_attrs:
        annotation = mapper.annotate(parse_type(ty), argument=False, owner=owner)
        body.append((name, [f"    {name}: ClassVar[{annotation}]"]))
    for variant in cls.variants:
        body.append((variant, [f"    {variant}: ClassVar[{owner}]"]))
    getters: dict[str, str] = {}
    setters: dict[str, str] = {}
    for name, ty, settable in cls.fields:
        parsed = parse_type(ty)
        getters[name] = mapper.annotate(parsed, argument=False, owner=owner)
        if settable:
            setters[name] = mapper.annotate(parsed, argument=True, owner=owner)
    for method in cls.methods:
        if has_attr(method.attrs, "getter"):
            getters[_accessor_name(method, "getter", "get_")] = _return(method, mapper, owner)
            continue
        if has_attr(method.attrs, "setter"):
            params, _ = python_params(method, mapper, owner, drop_first=False)
            (value,) = params
            setters[_accessor_name(method, "setter", "set_")] = value.annotation
            continue
        key = "__new__" if has_attr(method.attrs, "new") else method.name
        body.append((key, emit_method(method, cls, mapper)))
    for name in sorted(set(getters) | set(setters)):
        if name not in getters:
            raise ValueError(f"{cls.origin}: {owner}.{name} has a setter but no getter")
        entry = ["    @property", f"    def {name}(self) -> {getters[name]}: ..."]
        if name in setters:
            entry += [f"    @{name}.setter", f"    def {name}(self, value: {setters[name]}) -> None: ..."]
        body.append((name, entry))
    if any(method.name in _COMPARISONS for method in cls.methods):
        # Defining `__eq__` without `__hash__` makes CPython set `__hash__ = None`.
        if not any(method.name == "__hash__" for method in cls.methods):
            body.append(("__hash__", ["    __hash__: ClassVar[None]  # type: ignore[assignment]"]))
    body.sort(key=lambda entry: (entry[0] != "__new__", entry[0]))
    for _, entry in body:
        lines.extend(entry)
    if not body:
        lines.append("    ...")
    return lines


def emit_method(method: RustFn, cls: RustClass, mapper: TypeMapper) -> list[str]:
    owner = cls.python_name
    if has_attr(method.attrs, "new"):
        params, _ = python_params(method, mapper, owner, drop_first=False)
        return [f"    def __new__({render_params(params, 'cls')}) -> {owner}: ..."]
    decorators: list[str] = []
    receiver: str | None = "self"
    drop_first = False
    if has_attr(method.attrs, "staticmethod"):
        decorators.append("    @staticmethod")
        receiver = None
    elif has_attr(method.attrs, "classmethod"):
        decorators.append("    @classmethod")
        receiver = "cls"
        drop_first = True
    elif method.receiver is None:
        raise ValueError(f"{method.origin}: {owner}.{method.name} has no receiver")
    params, returns = python_params(method, mapper, owner, drop_first=drop_first)
    if _is_slot(method.name):
        for param in params:
            param.kind = "positional"
    if method.name in _COMPARISONS:
        # PyO3 returns NotImplemented for an operand it cannot extract.
        for param in params:
            param.annotation = "object"
    ret = returns or _return(method, mapper, owner)
    return decorators + [f"    def {method.name}({render_params(params, receiver)}) -> {ret}: ..."]


def _value_annotation(name: str, value: Sequence[Token], crate: Crate, mapper: TypeMapper) -> str:
    words = [t.text for t in value]
    if len(value) == 1 and value[0].kind == "str":
        return "str"
    if words[:2] == ["env", "!"]:
        return "str"
    if "get_type" in words:
        target = words[words.index("get_type") + 3]
        if target not in crate.exceptions:
            raise ValueError(f"{value[0].origin}: {name} registers unknown type {target}")
        return f"type[{target}]"
    if len(value) == 1 and value[0].text in crate.constants:
        return mapper.annotate(parse_type(crate.constants[value[0].text]), argument=False, owner=None)
    raise ValueError(f"{value[0].origin}: cannot type module value {name} = {text(value)!r}")


@dataclasses.dataclass
class Exports:
    """What ``gamfit._rust`` binds, split by kind."""

    classes: list[RustClass]  # every class a stub annotation can name
    registered_classes: set[str]  # the Python names the module binds
    exceptions: list[Exception_]
    values: list[tuple[str, str]]
    functions: list[RustFn]
    registered_names: set[str]  # every name the registration adds, `__doc__` included


def exports(crate: Crate, mapper: TypeMapper) -> Exports:
    reg = crate.registration
    registered = {crate.classes[name].python_name for name in reg.classes}
    # A class that is returned but never registered is still a real runtime
    # type; it is emitted so annotations resolve, under its Python name.
    classes = [
        cls
        for cls in sorted(crate.classes.values(), key=lambda c: c.python_name)
        if cls.python_name in registered or _referenced(cls, crate)
    ]
    exception_names: set[str] = set()
    values: list[tuple[str, str]] = []
    for name, value in reg.values:
        if name == "__doc__":
            continue
        annotation = _value_annotation(name, value, crate, mapper)
        if annotation.startswith("type[") and annotation[5:-1] == name:
            exception_names.add(name)
            continue
        values.append((name, annotation))
    exceptions = sorted(
        (e for e in crate.exceptions.values() if e.name in exception_names), key=lambda e: e.name
    )
    functions = [crate.functions[name] for name in sorted(set(reg.functions))]
    names = registered | {name for name, _ in reg.values} | {f.name for f in functions}
    return Exports(classes, registered, exceptions, sorted(values), functions, names)


def generate(crate: Crate) -> str:
    """The text of ``gamfit/_rust.pyi``."""
    mapper = TypeMapper(crate)
    found = exports(crate, mapper)
    sections: list[tuple[str, list[str]]] = [
        (cls.python_name, emit_class(cls, mapper)) for cls in found.classes
    ]
    for exception in found.exceptions:
        base = exception.base[2:] if exception.base.startswith("Py") else exception.base
        sections.append((exception.name, [f"class {exception.name}({base}): ..."]))
    for function in found.functions:
        sections.append((function.name, emit_function(function, mapper)))
    out = [HEADER]
    # PyO3 appends every name the module registers to its `__all__`.
    out.append("__all__ = [")
    out.extend(f'    "{name}",' for name in sorted(found.registered_names))
    out.append("]")
    out.append("")
    out.extend(f"{name}: {annotation}" for name, annotation in found.values)
    for _, lines in _order_exceptions(sections, crate):
        out.append("")
        out.extend(lines)
    return "\n".join(out) + "\n"


def generate_protocol(crate: Crate) -> str:
    """The text of ``gamfit/_rust_module.pyi``: the module's attributes as a Protocol."""
    mapper = TypeMapper(crate, qualifier="_rust.")
    found = exports(crate, mapper)
    members: list[tuple[str, list[str]]] = []
    for name, annotation in found.values:
        if annotation.startswith("type["):
            annotation = f"type[_rust.{annotation[5:-1]}]"
        members.append((name, [f"    {name}: {annotation}"]))
    for cls in found.classes:
        if cls.python_name in found.registered_classes:
            members.append((cls.python_name, [f"    {cls.python_name}: type[_rust.{cls.python_name}]"]))
    for exception in found.exceptions:
        members.append((exception.name, [f"    {exception.name}: type[_rust.{exception.name}]"]))
    for function in found.functions:
        members.append((function.name, ["    @staticmethod", *emit_function(function, mapper, "    ")]))
    out = [PROTOCOL_HEADER, "class RustModule(Protocol):"]
    for _, lines in sorted(members, key=lambda member: member[0]):
        out.extend(lines)
    return "\n".join(out) + "\n"


def _order_exceptions(
    sections: list[tuple[str, list[str]]], crate: Crate
) -> Iterator[tuple[str, list[str]]]:
    """Sort by name, but emit each exception after its base (stubs allow either,
    ordering by base keeps the file readable)."""
    remaining = sorted(sections, key=lambda entry: entry[0])
    done: set[str] = set()
    while remaining:
        for index, (name, lines) in enumerate(remaining):
            base = crate.exceptions[name].base if name in crate.exceptions else None
            if base is None or base not in crate.exceptions or base in done:
                done.add(name)
                yield remaining.pop(index)
                break
        else:
            raise ValueError("cyclic exception hierarchy")


def _referenced(cls: RustClass, crate: Crate) -> bool:
    names = {cls.rust_name}
    for function in crate.functions.values():
        if function.name in crate.registration.functions and names & {t.text for t in function.ret}:
            return True
    for other in crate.classes.values():
        for method in other.methods:
            if names & {t.text for t in method.ret}:
                return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if a stub is stale")
    options = parser.parse_args(argv)
    crate = parse_crate(SOURCE)
    outputs = {STUB: generate(crate), PROTOCOL: generate_protocol(crate)}
    if not options.check:
        for path, fresh in outputs.items():
            path.write_text(fresh)
        return 0
    stale = False
    for path, fresh in outputs.items():
        current = path.read_text() if path.exists() else ""
        if current == fresh:
            continue
        stale = True
        relative = str(path.relative_to(ROOT))
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            fresh.splitlines(keepends=True),
            fromfile=relative,
            tofile="generated",
        )
        sys.stderr.writelines(diff)
        sys.stderr.write(f"{relative} is stale; run `python scripts/gen_rust_stub.py`\n")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())

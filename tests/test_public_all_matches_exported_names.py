from __future__ import annotations

import ast
import importlib
import inspect
import typing
from types import ModuleType

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def _init_module_ast() -> ast.Module:
    return ast.parse(inspect.getsource(gamfit))


def _declared_submodules(tree: ast.Module) -> set[str]:
    """Submodules bound by a top-level ``from . import name`` in ``__init__``."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is None and node.level == 1:
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def test___all___matches_real_exports() -> None:
    exported = set(gamfit.__all__)
    missing = sorted(name for name in exported if not hasattr(gamfit, name))
    assert not missing, f"__all__ should not contain names that are not actually exported: {missing}"


def test___all___is_a_static_literal() -> None:
    # Type checkers only honour re-exports listed in a literal ``__all__``;
    # a computed one makes every ``gamfit.<name>`` "not explicitly exported".
    assignments = [
        node
        for node in _init_module_ast().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]
    assert len(assignments) == 1, "gamfit.__all__ must be assigned exactly once"
    value = assignments[0].value
    assert isinstance(value, ast.List), "gamfit.__all__ must be a list literal"
    assert all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in value.elts
    ), "gamfit.__all__ must contain only string literals"
    assert gamfit.__all__ == sorted(set(gamfit.__all__)), "gamfit.__all__ must be sorted and unique"


def test___all___equals_the_public_bindings() -> None:
    # Every public non-module binding plus the explicitly imported submodules;
    # modules that land in the namespace as import side effects stay private.
    submodules = _declared_submodules(_init_module_ast())
    public = {
        name
        for name, value in vars(gamfit).items()
        if not name.startswith("_") and (not isinstance(value, ModuleType) or name in submodules)
    }
    public.add("__version__")
    assert set(gamfit.__all__) == public, (
        f"missing from __all__: {sorted(public - set(gamfit.__all__))}; "
        f"stale in __all__: {sorted(set(gamfit.__all__) - public)}"
    )

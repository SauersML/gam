"""``docs/diagnostics.md`` must name exactly the kwargs ``validate_formula`` takes.

The prose list drifted silently once already. ``c7196cd9f`` deleted
``adaptive_regularization`` from every surface — the CLI flag, the wire key,
``FitConfig`` and ``FitOptions`` — but the ``validate_formula()`` section of
``docs/diagnostics.md`` kept advertising "adaptive regularization" among the
accepted keyword arguments. Because the unknown-option refusal fires on
unrecognised keys, the documented call did not merely do nothing: it raised.
The same paragraph had also never picked up ``persistent_warm_start_root``,
``noise_formula``, ``noise_offset`` or ``flexible_link``, so the drift ran in
both directions and no gate could see either.

This test reads both sides from source and compares the sets, so a kwarg added
to or removed from the signature fails here until the documented list follows.
It parses ``gamfit/_api.py`` with :mod:`ast` rather than importing ``gamfit``:
the contract is between a source signature and a source document, and the
in-tree ``gamfit`` package carries no compiled ``_rust``, so an import-based
check would be unrunnable exactly where this file lives.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_API = _REPO_ROOT / "gamfit" / "_api.py"
_DOCS = _REPO_ROOT / "docs" / "diagnostics.md"

# The paragraph is introduced by this sentence and runs to the blank line that
# ends it. Anchoring on the sentence rather than on a line number keeps the
# test attached to the claim, not to the file's current layout.
_INTRO = "Accepts these parser/materialization keyword arguments from `gamfit.fit`,"

# Named in the docs immediately after the list as the things validation
# refuses. They must NOT appear in the signature.
_FIT_ONLY = (
    "constraints",
    "latents",
    "penalties",
    "smooths",
    "precision_hyperpriors",
    "response_geometry",
)


def _signature_kwargs() -> list[str]:
    """The keyword-only parameters of ``gamfit._api.validate_formula``."""
    tree = ast.parse(_API.read_text(encoding="utf-8"), filename=str(_API))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate_formula":
            return [arg.arg for arg in node.args.kwonlyargs]
    raise AssertionError(f"no top-level `def validate_formula` in {_API}")


def _documented_kwargs() -> list[str]:
    """The backticked names in the ``validate_formula`` accepted-kwargs list."""
    text = _DOCS.read_text(encoding="utf-8")
    start = text.find(_INTRO)
    assert start != -1, (
        f"{_DOCS} no longer contains the sentence this test is anchored to: "
        f"{_INTRO!r}. Update the anchor and the list together."
    )
    end = text.find("\n\n", start)
    assert end != -1, f"unterminated accepted-kwargs paragraph in {_DOCS}"
    paragraph = text[start:end]
    # Drop the intro sentence itself: it backticks `gamfit.fit`, which is not a
    # kwarg.
    listed = paragraph[len(_INTRO) :]
    return re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", listed)


def test_validate_formula_docs_list_is_the_signature() -> None:
    signature = _signature_kwargs()
    documented = _documented_kwargs()

    assert len(documented) == len(set(documented)), (
        "the documented kwarg list repeats a name: "
        f"{sorted({n for n in documented if documented.count(n) > 1})}"
    )

    missing_from_docs = sorted(set(signature) - set(documented))
    not_in_signature = sorted(set(documented) - set(signature))
    assert not missing_from_docs and not_in_signature == [], (
        "docs/diagnostics.md and validate_formula()'s signature disagree.\n"
        f"  accepted but undocumented: {missing_from_docs}\n"
        f"  documented but not accepted: {not_in_signature}"
    )

    # Same order in both places, so a reader can check them line by line.
    assert documented == signature, (
        "the documented kwargs are the right set in the wrong order; list them "
        "in signature order.\n"
        f"  signature: {signature}\n"
        f"  documented: {documented}"
    )


def test_validate_formula_refuses_the_fit_only_objects_the_docs_name() -> None:
    signature = set(_signature_kwargs())
    leaked = sorted(name for name in _FIT_ONLY if name in signature)
    assert leaked == [], (
        "docs/diagnostics.md says validate_formula refuses these fit-only "
        f"objects, but they are in its signature: {leaked}"
    )

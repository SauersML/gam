"""An overcomplete (K > P) payload must reach the parser that can read it.

#2567 filed three facts. Two were repaired in the loader: the ``ManifoldSAE``
pyclass name collision, and ``gamfit.load``'s sniff, which now dispatches on the
*exact* schema tag rather than a substring of it. The third stayed live, because
the repair landed inside ``gamfit.load`` while #2502's flagship readers --
``experiments/i2502_overcomplete_manifold/{splice_eval,interp_atoms,steer_flagship}.py``
-- never call it. They hold a pickled ``to_dict()`` payload, not a path, and
reached straight into ``ManifoldSAE.from_dict``, which is pinned to ``/v8``.

The symptom is worth stating because it is not obviously a routing bug: a
support payload handed to the dense parser fails on a *dense* field name
(``missing field "assignment"``), so it reads as a corrupt file rather than as a
misrouted one. The splice, interpretation, and steering benchmarks could not read
the very dictionaries they exist to measure.

The dispatch is now ``gamfit.model_from_dict``, and these tests pin the routing
itself: which parser claims the payload, identified by the parser that reports
the failure. Deliberately fit-free -- minting a real overcomplete fit costs a
support-lane solve that does not converge on small synthetic data, and the defect
under test is the routing, not the fit.
"""

from __future__ import annotations

import importlib
import json
import pickle
from typing import Any

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit

# Each parser names itself in its own errors, which makes the parser that
# claimed a payload observable without constructing a valid model.
SUPPORT_PARSER = "ManifoldSAESupport.from_dict"
DENSE_PARSER = "ManifoldSAE.from_json"

SUPPORT_PAYLOAD = {"schema": "gamfit.ManifoldSAE/support-v2"}
DENSE_PAYLOAD = {"schema": "gamfit.ManifoldSAE/v10"}


def _claiming_parser(fn: Any, payload: Any) -> str:
    """Return the message of whichever parser claimed (and rejected) the payload."""
    with pytest.raises(Exception) as excinfo:
        fn(payload)
    return str(excinfo.value)


def test_support_tag_routes_to_the_support_parser() -> None:
    message = _claiming_parser(gamfit.model_from_dict, SUPPORT_PAYLOAD)
    assert SUPPORT_PARSER in message, (
        "an overcomplete payload must be claimed by the support parser; "
        f"instead it was claimed by: {message}"
    )
    assert DENSE_PARSER not in message


def test_dense_tag_still_routes_to_the_dense_parser() -> None:
    message = _claiming_parser(gamfit.model_from_dict, DENSE_PAYLOAD)
    assert DENSE_PARSER in message, (
        f"a dense payload must still reach the dense parser; got: {message}"
    )


def test_unknown_tag_falls_back_to_the_dense_parser() -> None:
    """Only the exact support tag diverts; anything else keeps its old destination."""
    message = _claiming_parser(gamfit.model_from_dict, {"schema": "unknown/v0"})
    assert DENSE_PARSER in message


def test_the_old_direct_call_still_misroutes_the_support_payload() -> None:
    """Negative control: the switch changes the destination, it is not decoration.

    If ``ManifoldSAE.from_dict`` ever began accepting support payloads, every
    assertion above would pass through either branch and would stop proving that
    the readers were re-pointed. This pins the two destinations as distinct, and
    documents the exact symptom #2567 filed.
    """
    message = _claiming_parser(gamfit.ManifoldSAE.from_dict, SUPPORT_PAYLOAD)
    assert DENSE_PARSER in message
    assert SUPPORT_PARSER not in message


def test_pickle_round_trip_preserves_the_tag() -> None:
    """The readers carry the payload through pickle; the tag must survive it."""
    restored = pickle.loads(pickle.dumps(SUPPORT_PAYLOAD, protocol=4))
    assert restored["schema"] == gamfit.SUPPORT_SAE_SCHEMA
    message = _claiming_parser(gamfit.model_from_dict, restored)
    assert SUPPORT_PARSER in message


def test_load_and_model_from_dict_route_identically(tmp_path: Any) -> None:
    """``gamfit.load`` and ``gamfit.model_from_dict`` are one dispatch, not two."""
    path = tmp_path / "overcomplete.json"
    path.write_text(json.dumps(SUPPORT_PAYLOAD), encoding="utf-8")

    from_disk = _claiming_parser(gamfit.load, str(path))
    from_payload = _claiming_parser(gamfit.model_from_dict, SUPPORT_PAYLOAD)

    assert SUPPORT_PARSER in from_disk
    assert from_disk == from_payload, (
        "the on-disk and in-memory routes must reach the same parser: "
        f"disk said {from_disk!r}, payload said {from_payload!r}"
    )

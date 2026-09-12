#!/usr/bin/env python3
"""Strict paired Eq-4 comparison for GitHub #2283.

The input must contain exactly one authoritative ``external_topk`` result and one
authoritative resumed ``hybrid_rust`` result for ``--run-id``. Both rows must carry
the same canonical pair identity: code/wheel/scorer hashes, shard manifest, exact
sample/split/bits row hashes, and the shared measurement configuration. Duplicate,
stale, or mismatched rows are errors rather than freshness heuristics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys


# v2 (#2283): dictionary bits are charged against a DECLARED amortization_horizon
# that is separate from the bits estimation subsample. v1 rows priced the
# dictionary term with the confounded subsample N and are NOT comparable, so this
# comparator rejects them at the schema gate and requires an explicit horizon.
PAIR_SCHEMA = "gam.issue2283.eq4-pair.v2"
ARMS = ("external_topk", "hybrid_rust")
TARGETS = ("0.99", "0.95", "0.9", "0.8")


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_sha256(payload) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def load(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


def _authoritative_pair(records, run_id):
    selected = {
        arm: [
            record
            for record in records
            if record.get("record_type") == "result"
            and record.get("run_id") == run_id
            and record.get("arm") == arm
        ]
        for arm in ARMS
    }
    for arm, rows in selected.items():
        if len(rows) != 1:
            raise ValueError(
                f"run_id={run_id!r} requires exactly one {arm!r} result; found {len(rows)}"
            )
    external, hybrid = (selected[arm][0] for arm in ARMS)
    for record in (external, hybrid):
        if record.get("issue") != 2283:
            raise ValueError(f"{record['arm']} row is not an issue-2283 result")
        identity = record.get("pair_identity")
        if not isinstance(identity, dict) or identity.get("schema") != PAIR_SCHEMA:
            raise ValueError(f"{record['arm']} row has no canonical pair identity")
        if identity.get("run_id") != run_id:
            raise ValueError(f"{record['arm']} pair identity has the wrong run ID")
        digest = _payload_sha256(identity)
        if record.get("pair_identity_sha256") != digest:
            raise ValueError(f"{record['arm']} pair identity digest is invalid")
        data = identity["data"]
        if record.get("data_manifest", {}).get("sha256") != data["manifest_sha256"]:
            raise ValueError(f"{record['arm']} shard manifest does not match its pair identity")
        if record.get("bits_test_positions_sha256") != data["bits_test_positions_sha256"]:
            raise ValueError(f"{record['arm']} bit positions do not match its pair identity")
        if record.get("bits_row_ids_sha256") != data["bits_row_ids_sha256"]:
            raise ValueError(f"{record['arm']} bit rows do not match its pair identity")
        if record.get("bits_dict_params_faithful") is not True:
            raise ValueError(f"{record['arm']} row is not dictionary-parameter faithful")
        horizon = record.get("bits_amortization_horizon")
        if not isinstance(horizon, int) or horizon < 2:
            raise ValueError(
                f"{record['arm']} row has no declared amortization horizon (>= 2); "
                "it predates the #2283 confound fix and must be re-scored"
            )
        if record.get("pair_identity", {}).get("config", {}).get(
            "amortization_horizon"
        ) != horizon:
            raise ValueError(
                f"{record['arm']} amortization horizon disagrees with its pair identity"
            )
    if external["pair_identity"] != hybrid["pair_identity"]:
        raise ValueError("external and hybrid rows have different config/provenance identities")
    if hybrid.get("hybrid_phase") != "curved-resume":
        raise ValueError("hybrid result was not produced by the curved-resume phase")
    if not hybrid.get("flat_checkpoint_sha256"):
        raise ValueError("hybrid result has no flat checkpoint digest")
    return external, hybrid


def _components(record, target):
    components = {
        "support": record[f"bits_support_bits"],
        "code": record[f"bits_code_bits_at_r2_{target}"],
        "residual": record[f"bits_resid_bits_at_r2_{target}"],
        "dictionary": record["bits_dictionary_bits"],
    }
    total = record[f"bits_bits_at_r2_{target}"]
    if not math.isclose(total, sum(components.values()), rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(
            f"{record['arm']} R2={target} total does not reconcile with its four components"
        )
    return total, components


def _predicted_support_margin(hybrid) -> float:
    """`log2 C(G, L0)` of the external config minus that of the hybrid config.

    Uses the same overflow-safe product form the Rust scorer's `selection_bits`
    uses, so this is the scorer's own arithmetic rather than an approximation of
    it. A chart firing spends `1 + d` of the active-scalar budget where a flat
    firing spends one, so the hybrid names fewer atoms per token while
    transmitting the same number of scalars; that is the whole predicted effect.
    """
    # Imported here, against this file's own directory: the comparator's
    # regression test loads this module by path with `spec_from_file_location`
    # and never puts the experiment directory on `sys.path`.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from crossover_theorem_check import selection_bits

    flat_actives = int(hybrid["top_k"]) - int(hybrid["curved_k"]) * (1 + int(hybrid["d_atom"]))
    external = selection_bits(int(hybrid["K"]), int(hybrid["top_k"]))
    paired = selection_bits(
        int(hybrid["k_flat"]) + int(hybrid["curved_atoms"]),
        flat_actives + int(hybrid["curved_k"]),
    )
    return external - paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    external, hybrid = _authoritative_pair(load(args.results), args.run_id)
    print(
        f"=== issue #2283 paired Eq-4 run {args.run_id} "
        f"identity={external['pair_identity_sha256']} ==="
    )
    print(f"EV external={external['ev']:.6f} hybrid={hybrid['ev']:.6f}")
    print(
        f"{'target R2':>10} {'external':>12} {'hybrid':>12} {'delta':>12}  "
        "(support/code/residual/dictionary)"
    )
    for target in TARGETS:
        external_total, external_parts = _components(external, target)
        hybrid_total, hybrid_parts = _components(hybrid, target)
        delta = {
            name: external_parts[name] - hybrid_parts[name]
            for name in external_parts
        }
        print(
            f"{target:>10} {external_total:>12.1f} {hybrid_total:>12.1f} "
            f"{external_total-hybrid_total:>+12.1f}  "
            f"({delta['support']:+.1f}/{delta['code']:+.1f}/"
            f"{delta['residual']:+.1f}/{delta['dictionary']:+.1f})"
        )

    external_099, external_099_parts = _components(external, "0.99")
    hybrid_099, hybrid_099_parts = _components(hybrid, "0.99")

    # The theorem's claim for this configuration, in closed form and with no
    # fitted quantity in it: the faithful k_flat config equalises the decoder
    # scalar counts exactly, so Ddict = 0; a circle-class chart has span
    # s = d + 1, so the crossover code term (s - d - 1)/2 * log2(lambda/delta) is
    # identically zero. What is left is the support term of the two
    # CONFIGURATIONS. Printing it next to the measured margin is what makes the
    # verdict readable: a margin that is not the support term is some other
    # effect wearing the theorem's name.
    predicted = _predicted_support_margin(hybrid)
    print(
        f"theorem @R2=0.99: Ddict 0.000 (measured {external_099_parts['dictionary'] - hybrid_099_parts['dictionary']:+.3f})"
        f" · Dcode 0.000 (measured {external_099_parts['code'] - hybrid_099_parts['code']:+.3f})"
        f" · Dsupport {predicted:+.3f} (measured {external_099_parts['support'] - hybrid_099_parts['support']:+.3f})"
    )
    for row in (external, hybrid):
        audit = row.get("bits_faithfulness_audit")
        currency = None if audit is None else audit.get("support_currency")
        if currency is not None:
            print(
                f"  {row['arm']:>13} support currency: combinatorial "
                f"{currency['combinatorial_bits']:.3f} vs independent "
                f"{currency['independent_support_bits']:.3f} "
                f"(gap {currency['combinatorial_bits'] - currency['independent_support_bits']:+.3f}, "
                f"{currency['atoms_that_ever_fire']}/{currency['atoms']} atoms ever fire)"
            )

    if hybrid_099 >= external_099:
        raise ValueError(
            f"#2283 failed: hybrid {hybrid_099:.6f} does not beat paired external "
            f"{external_099:.6f}"
        )
    print(f"PASS R2=0.99: hybrid={hybrid_099:.6f} < paired={external_099:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

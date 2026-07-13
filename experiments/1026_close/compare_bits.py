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


# v2 (#2283): dictionary bits are charged against a DECLARED amortization_horizon
# that is separate from the bits estimation subsample. v1 rows priced the
# dictionary term with the confounded subsample N and are NOT comparable, so this
# comparator rejects them at the schema gate and requires an explicit horizon.
PAIR_SCHEMA = "gam.issue2283.eq4-pair.v2"
ARMS = ("external_topk", "hybrid_rust")
TARGETS = ("0.99", "0.95", "0.9", "0.8")
HISTORICAL_BAR_R2_099 = 56_322.0


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

    external_099, _ = _components(external, "0.99")
    hybrid_099, _ = _components(hybrid, "0.99")
    if hybrid_099 >= external_099:
        raise ValueError(
            f"#2283 failed: hybrid {hybrid_099:.6f} does not beat paired external "
            f"{external_099:.6f}"
        )
    if hybrid_099 >= HISTORICAL_BAR_R2_099:
        raise ValueError(
            f"#2283 failed: hybrid {hybrid_099:.6f} does not beat historical bar "
            f"{HISTORICAL_BAR_R2_099:.0f}"
        )
    print(
        f"PASS R2=0.99: hybrid={hybrid_099:.6f} < paired={external_099:.6f} "
        f"and < historical={HISTORICAL_BAR_R2_099:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Strict paired Eq-4 measurement for GitHub #2283.

The input must contain exactly one authoritative result for ``--run-id`` from each
of ``external_topk``, ``gam_flat`` and the resumed ``hybrid_rust``. All three rows
must carry the same canonical pair identity: code/wheel/scorer hashes, shard
manifest, exact sample/split/bits row hashes, and the shared measurement
configuration. Duplicate, stale, or mismatched rows are errors rather than
freshness heuristics.

``gam_flat`` is the paired control: the hybrid's own trainer at the external
bar's ``K`` and ``top_k``. The hybrid differs from it only in replacing flat atoms
with curved charts at an equal dictionary count and an equal active-scalar budget,
which is the difference the crossover theorem is about. ``external_topk`` is
reported beside it; its gap to the hybrid also contains the gap between two
trainers.

For each difference the comparator prints the plug-in value, the grouped-jackknife
bias-corrected value and its standard error (``eq4_jackknife``), and the verdict at
R2 = 0.99. It exits 0 on any verdict: the measurement is the result (#2283).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys


# v3 (#2283): the measurement pairs the hybrid with gam_flat, and every row
# carries the grouped jackknife over its scored rows. v2 rows have neither, so
# this comparator rejects them at the schema gate.
PAIR_SCHEMA = "gam.issue2283.eq4-pair.v3"
EXTERNAL, CONTROL, HYBRID = "external_topk", "gam_flat", "hybrid_rust"
ARMS = (EXTERNAL, CONTROL, HYBRID)
TARGETS = ("0.99", "0.95", "0.9", "0.8")
VERDICT_TARGET = "0.99"
# The verdict is a two-sided interval of this many standard errors, about 95 %
# coverage for an approximately normal estimate.
VERDICT_STANDARD_ERRORS = 2.0
COMPONENTS = ("support", "code", "residual", "dictionary")


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_sha256(payload) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sibling(name: str):
    """Import a module from this file's own directory.

    The comparator's regression test loads this module by path with
    `spec_from_file_location` and never puts the experiment directory on
    `sys.path`.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return __import__(name)


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
    """The external, control and hybrid rows of ``run_id``, in ``ARMS`` order."""
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
    rows = tuple(selected[arm][0] for arm in ARMS)
    for record in rows:
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
        if identity.get("config", {}).get("amortization_horizon") != horizon:
            raise ValueError(
                f"{record['arm']} amortization horizon disagrees with its pair identity"
            )
        _leave_one_out(record)
    external, control, hybrid = rows
    if not (external["pair_identity"] == control["pair_identity"] == hybrid["pair_identity"]):
        raise ValueError("the three rows have different config/provenance identities")
    if hybrid.get("hybrid_phase") != "curved-resume":
        raise ValueError("hybrid result was not produced by the curved-resume phase")
    if not hybrid.get("flat_checkpoint_sha256"):
        raise ValueError("hybrid result has no flat checkpoint digest")
    for field in ("sparse_score_mode", "sparse_minibatch"):
        if control.get(field) is None or control.get(field) != hybrid.get(field):
            raise ValueError(
                f"gam_flat and the hybrid flat tier must share one trainer setting; "
                f"{field} is {control.get(field)!r} and {hybrid.get(field)!r}"
            )
    partitions = {record["bits_jackknife"]["block_positions"] for record in rows}
    if len(partitions) != 1:
        raise ValueError("the three rows were jackknifed over different row blocks")
    return rows


def _leave_one_out(record):
    """The row's deleted-block scorer outputs, checked against its pair identity."""
    jackknife = record.get("bits_jackknife")
    groups = record["pair_identity"]["config"].get("jackknife_groups")
    if not isinstance(jackknife, dict) or not isinstance(groups, int) or groups < 2:
        raise ValueError(f"{record['arm']} row carries no grouped jackknife")
    if jackknife.get("groups") != groups or len(jackknife.get("leave_one_out", [])) != groups:
        raise ValueError(
            f"{record['arm']} jackknife does not have the identity's {groups} deleted blocks"
        )
    return jackknife["leave_one_out"]


def _components(record, target, prefix="bits_"):
    """Total bits at ``target`` and its four components, which must reconcile.

    ``prefix`` is ``"bits_"`` for a result row and ``""`` for one deleted-block
    scorer record.
    """
    components = {
        "support": record[f"{prefix}support_bits"],
        "code": record[f"{prefix}code_bits_at_r2_{target}"],
        "residual": record[f"{prefix}resid_bits_at_r2_{target}"],
        "dictionary": record[f"{prefix}dictionary_bits"],
    }
    total = record[f"{prefix}bits_at_r2_{target}"]
    if not math.isclose(total, sum(components.values()), rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(
            f"{record.get('arm', 'deleted-block')} R2={target} total does not reconcile "
            "with its four components"
        )
    return total, components


def _parts(record, target):
    """``{"total", *COMPONENTS}`` -> (full value, deleted-block values)."""
    total, components = _components(record, target)
    deleted = [_components(entry, target, prefix="") for entry in _leave_one_out(record)]
    parts = {"total": (total, [entry_total for entry_total, _ in deleted])}
    for name in COMPONENTS:
        parts[name] = (components[name], [entry[name] for _, entry in deleted])
    return parts


def paired_difference(hybrid, other, target):
    """``B(hybrid) - B(other)`` at ``target``, total and per component."""
    jackknife = _sibling("eq4_jackknife")
    mine, theirs = _parts(hybrid, target), _parts(other, target)
    return {
        name: jackknife.paired_jackknife(
            mine[name][0], theirs[name][0], mine[name][1], theirs[name][1]
        )
        for name in ("total", *COMPONENTS)
    }


def verdict(estimate) -> str:
    """The #2283 verdict on one bias-corrected difference ``B(hybrid) - B(other)``."""
    reach = VERDICT_STANDARD_ERRORS * estimate.standard_error
    if estimate.corrected + reach < 0.0:
        return "hybrid shorter"
    if estimate.corrected - reach > 0.0:
        return "hybrid longer"
    return "unresolved at this n"


def _predicted_support_margin(hybrid) -> float:
    """The flat configs' support code minus the hybrid config's.

    Each config names a fixed number of atoms on every row, so the scorer's
    cardinality-then-subset code `log2(G+1) + mean_i log2 C(G, |S_i|)` is
    `log2(G+1) + log2 C(G, L0)` for it, with no fitted input. A chart firing
    spends `1 + d` of the active-scalar budget where a flat firing spends one, so
    the hybrid names fewer atoms per token while transmitting the same number of
    scalars; that is the whole predicted effect.
    """
    support_code_bits = _sibling("crossover_theorem_check").support_code_bits
    flat_actives = int(hybrid["top_k"]) - int(hybrid["curved_k"]) * (1 + int(hybrid["d_atom"]))
    flat = support_code_bits(int(hybrid["K"]), int(hybrid["top_k"]))
    paired = support_code_bits(
        int(hybrid["k_flat"]) + int(hybrid["curved_atoms"]),
        flat_actives + int(hybrid["curved_k"]),
    )
    return flat - paired


def _format(estimate) -> str:
    return (
        f"{estimate.plug_in:+10.3f} -> {estimate.corrected:+10.3f} "
        f"± {estimate.standard_error:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    jackknife = _sibling("eq4_jackknife")

    external, control, hybrid = _authoritative_pair(load(args.results), args.run_id)
    groups = hybrid["pair_identity"]["config"]["jackknife_groups"]
    print(
        f"=== issue #2283 paired Eq-4 run {args.run_id} "
        f"identity={hybrid['pair_identity_sha256']} jackknife groups={groups} ==="
    )
    print(
        f"EV external={external['ev']:.6f} gam_flat={control['ev']:.6f} "
        f"hybrid={hybrid['ev']:.6f}"
    )
    print(
        "Delta = B(hybrid) - B(other), bits/token: plug-in -> bias-corrected ± SE"
    )
    for target in TARGETS:
        totals = {row["arm"]: _components(row, target)[0] for row in (external, control, hybrid)}
        against_control = paired_difference(hybrid, control, target)
        against_external = paired_difference(hybrid, external, target)
        print(
            f"R2={target:>4}  external {totals[EXTERNAL]:10.3f}  gam_flat "
            f"{totals[CONTROL]:10.3f}  hybrid {totals[HYBRID]:10.3f}"
        )
        for label, estimates in (("vs gam_flat", against_control), ("vs external", against_external)):
            print(f"  {label:<11} total      {_format(estimates['total'])}")
            for name in COMPONENTS:
                print(f"  {'':<11} {name:<10} {_format(estimates[name])}")

    n_rows = int(hybrid["bits_rows"])
    p_out = int(hybrid["p"])
    print(
        f"residual-term plug-in bias at R2={VERDICT_TARGET} (jackknife) vs the Wishart "
        f"all-active bound {jackknife.wishart_all_active_bias_bits(p_out, n_rows):+.3f} "
        f"(P={p_out}, n={n_rows}):"
    )
    for row in (external, control, hybrid):
        full, deleted = _parts(row, VERDICT_TARGET)["residual"]
        print(f"  {row['arm']:>13} {jackknife.jackknife(full, deleted).bias:+10.3f}")

    # The theorem's claim for this configuration, in closed form and with no
    # fitted quantity in it: the faithful k_flat config equalises the decoder
    # scalar counts exactly, so Ddict = 0; a circle-class chart has span
    # s = d + 1, so the crossover code term (s - d - 1)/2 * log2(lambda/delta) is
    # identically zero. What is left is the support term of the two
    # CONFIGURATIONS. A measured margin that is not the support term is some
    # other effect wearing the theorem's name.
    predicted = _predicted_support_margin(hybrid)
    measured = paired_difference(hybrid, control, VERDICT_TARGET)
    print(
        f"theorem @R2={VERDICT_TARGET} vs gam_flat: Ddict 0.000 (measured "
        f"{measured['dictionary'].plug_in:+.3f}) · Dcode 0.000 (measured "
        f"{measured['code'].plug_in:+.3f}) · Dsupport {-predicted:+.3f} (measured "
        f"{measured['support'].plug_in:+.3f})"
    )
    for row in (external, control, hybrid):
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

    for label, other in (("gam_flat", control), ("external", external)):
        estimate = paired_difference(hybrid, other, VERDICT_TARGET)["total"]
        print(
            f"VERDICT R2={VERDICT_TARGET} vs {label}: {verdict(estimate)} "
            f"(Delta {estimate.corrected:+.3f} ± {estimate.standard_error:.3f}, "
            f"{VERDICT_STANDARD_ERRORS:g} SE)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

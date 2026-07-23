#!/usr/bin/env python3
"""Emit publishable workspace crates in deterministic dependency-first order."""

from __future__ import annotations

import json
import sys


def fail(message: str) -> None:
    print(f"workspace publish order: {message}", file=sys.stderr)
    raise SystemExit(1)


metadata = json.load(sys.stdin)
workspace_ids = set(metadata["workspace_members"])
workspace_packages = {
    package["name"]: package
    for package in metadata["packages"]
    if package["id"] in workspace_ids
}
publishable = {
    name: package
    for name, package in workspace_packages.items()
    if package.get("publish") != []
}

dependencies: dict[str, set[str]] = {name: set() for name in publishable}
for name, package in publishable.items():
    for dependency in package["dependencies"]:
        if dependency["kind"] == "dev" or dependency.get("path") is None:
            continue
        dependency_name = dependency["name"]
        if dependency_name not in workspace_packages:
            fail(
                f"{name} has path dependency {dependency_name}, "
                "but it is not a workspace member"
            )
        if dependency_name not in publishable:
            fail(
                f"publishable crate {name} has non-dev path dependency "
                f"{dependency_name}, which has publish=false"
            )
        dependencies[name].add(dependency_name)

remaining = {name: set(required) for name, required in dependencies.items()}
order: list[str] = []
while remaining:
    ready = sorted(name for name, required in remaining.items() if not required)
    if not ready:
        cycle = ", ".join(
            f"{name}->[{', '.join(sorted(required))}]"
            for name, required in sorted(remaining.items())
        )
        fail(f"normal/build path-dependency cycle among publishable crates: {cycle}")
    order.extend(ready)
    for name in ready:
        del remaining[name]
    for required in remaining.values():
        required.difference_update(ready)

print("\n".join(order))

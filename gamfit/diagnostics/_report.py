"""Structured identifiability report.

A diagnostic that "checks an identifiability theorem" returns an
:class:`IdentifiabilityReport`. The report records, for one named theorem:

* ``name`` — short name of the diagnostic (e.g. ``"aux_richness"``).
* ``theorem`` — the citation / theorem the diagnostic is checking.
* ``preconditions`` — dict mapping each precondition string to ``True`` /
  ``False``. A passing precondition is ``True``.
* ``violations`` — list of human-readable strings, one per failed
  precondition.
* ``recommendations`` — list of concrete remediations, one per violation.
  ``len(recommendations) == len(violations)`` is the contract.
* ``passes`` — derived: ``len(violations) == 0``.

The ``__repr__`` renders a compact green-check / warning block usable from
notebooks and REPLs without external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IdentifiabilityReport:
    """Result of a single identifiability diagnostic.

    See module docstring for the field contract.
    """

    name: str
    theorem: str
    preconditions: dict[str, bool] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        """``True`` iff no precondition is violated."""
        return len(self.violations) == 0

    def __repr__(self) -> str:
        head = (
            f"[OK] {self.name}: identifiable ({self.theorem})"
            if self.passes
            else f"[WARN] {self.name}: {len(self.violations)} violation(s) ({self.theorem})"
        )
        if self.passes:
            return head
        body = []
        for v, r in zip(self.violations, self.recommendations):
            body.append(f"  - {v}")
            body.append(f"    -> {r}")
        return head + "\n" + "\n".join(body)


@dataclass(slots=True)
class CompositeIdentifiabilityReport:
    """Aggregate of multiple per-theorem :class:`IdentifiabilityReport`s.

    Returned by :func:`gamfit.diagnostics.identifiability_report`.
    """

    model_kind: str
    reports: list[IdentifiabilityReport] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """``True`` iff every contained report passes."""
        return all(r.passes for r in self.reports)

    def summary(self) -> str:
        """One-line summary of the composite report.

        Returns a sentence either confirming identifiability or naming the
        weakest theorem-class still satisfied. The intent is to be
        descriptive about *what* identifiability remains, not just to fail.
        """
        if not self.reports:
            return f"Identifiability: no applicable diagnostic for model kind {self.model_kind!r}."
        if self.passes:
            return "Identified."
        failed = [r.name for r in self.reports if not r.passes]
        if "anchor_consistency" in failed and len(failed) == 1:
            return "Identifiable up to a linear transform of the atom space (anchor consistency failed)."
        if "aux_richness" in failed:
            return "Not identified: auxiliary covariate insufficient."
        if "jacobian_sparsity" in failed:
            return "Not identified: decoder Jacobian sparsity precondition violated."
        return f"Not identified: {', '.join(failed)} failed."

    def detail(self) -> str:
        """Full per-check breakdown, one block per contained report."""
        lines = [f"Identifiability report for model kind: {self.model_kind}", ""]
        for r in self.reports:
            lines.append(repr(r))
            lines.append("")
        lines.append(self.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.detail()

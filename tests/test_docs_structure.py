"""Structural contracts for the published documentation.

``test_documentation_examples.py`` executes the code in the docs; this file
checks what a reader sees around that code: tables render as tables, the
engineering ledgers stay out of the site, the pages for pyGAM users are
reachable, and no runnable example grid-searches a smoothing parameter or a
basis size.
"""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
READMES = (ROOT / "README.md", ROOT / "README_PYPI.md")
PAGES = (*READMES, *sorted(DOCS_DIR.glob("*.md")))
FENCE_LINE = re.compile(r"^\s*```")
SEPARATOR = re.compile(r"^\|\s*:?-{3,}")
EXECUTABLE_PYTHON = re.compile(r"^```python[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


MKDOCS = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


def _exclude_patterns() -> list[str]:
    block = re.search(r"^exclude_docs: \|\n((?:[ \t]+\S.*\n)+)", MKDOCS, re.MULTILINE)
    assert block, "mkdocs.yml has no exclude_docs block"
    return [line.strip() for line in block[1].splitlines()]


def _nav_pages() -> list[str]:
    nav = re.search(r"^nav:\n((?:[ \t]+.*\n)+)", MKDOCS, re.MULTILINE)
    assert nav, "mkdocs.yml has no nav"
    return re.findall(r"^\s*- (?:[^:\n]+: )?(\S+\.md)\s*$", nav[1], re.MULTILINE)


def _excluded(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def _table_cell_count(line: str) -> int:
    # Pipes inside code spans or escaped as ``\|`` do not separate cells.
    plain = re.sub(r"`[^`]*`", "code", line.replace(r"\|", "pipe"))
    return plain.strip().strip("|").count("|") + 1


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    in_fence = False
    lines = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if FENCE_LINE.match(line):
            in_fence = not in_fence
            lines.append((number, ""))
        else:
            lines.append((number, "" if in_fence else line))
    return lines


@pytest.mark.parametrize("path", PAGES, ids=lambda path: str(path.relative_to(ROOT)))
def test_markdown_tables_are_well_formed(path):
    """Every table starts after a blank line, has a separator row and a fixed width.

    A table row that continues onto an unpiped line, or that follows a line of
    prose, is rendered as a paragraph of pipes instead of a table.
    """
    lines = _prose_lines(path)
    problems = []
    for index, (number, line) in enumerate(lines):
        if not line.startswith("|"):
            continue
        previous = lines[index - 1][1] if index else ""
        if previous.startswith("|"):
            continue
        following = lines[index + 1][1] if index + 1 < len(lines) else ""
        if previous.strip() or not SEPARATOR.match(following):
            problems.append(f"{number}: table header is not preceded by a blank line and followed by a separator")
            continue
        width = _table_cell_count(line)
        for row_number, row in lines[index + 1:]:
            if not row.startswith("|"):
                if row.strip():
                    problems.append(f"{row_number}: text directly after a table row: {row[:60]!r}")
                break
            if _table_cell_count(row) != width:
                problems.append(f"{row_number}: {_table_cell_count(row)} cells in a {width}-column table")
    assert not problems, f"{path.relative_to(ROOT)}:\n" + "\n".join(problems)


def test_engineering_ledgers_are_excluded_from_the_site():
    patterns = _exclude_patterns()
    ledgers = [
        path.name for path in DOCS_DIR.iterdir()
        if path.name.startswith(("issue-", "test-census-", "test-integrity", "public-api-"))
    ]
    assert ledgers, "the docs directory no longer carries ledgers; drop this test"
    published = [name for name in ledgers if not _excluded(name, patterns)]
    assert not published, f"ledgers would be published on the docs site: {published}"


def test_pages_for_pygam_users_are_reachable():
    patterns = _exclude_patterns()
    nav = set(_nav_pages())
    missing_files = sorted(page for page in nav if not (DOCS_DIR / page).is_file())
    assert not missing_files, f"nav names pages that do not exist: {missing_files}"
    for page in ("tour.md", "benchmarks.md"):
        assert page in nav, f"{page} is not in the mkdocs nav"
    for page in ("tour.md", "benchmarks.md"):
        assert f"({page})" in (DOCS_DIR / "README.md").read_text(encoding="utf-8"), page
    assert not [page for page in nav if _excluded(page, patterns)]


@pytest.mark.parametrize("path", PAGES, ids=lambda path: str(path.relative_to(ROOT)))
def test_no_executable_example_grid_searches(path):
    """Smoothing parameters come from REML/LAML, so no runnable example grid-searches.

    The migration guide quotes pyGAM's ``gridsearch()`` in a ``python no-exec``
    fence; that is a description of the other library, not an example.
    """
    for match in EXECUTABLE_PYTHON.finditer(path.read_text(encoding="utf-8")):
        body = match["body"]
        assert "GridSearchCV" not in body and "gridsearch(" not in body, body[:200]


@pytest.mark.parametrize("path", READMES, ids=lambda path: path.name)
def test_readme_opens_with_pitch_example_and_linked_claims(path):
    """The first screen is a pitch, a runnable example, then evidence-linked claims."""
    text = path.read_text(encoding="utf-8")
    first_fence = text.index("```python")
    claims_start = text.index("- **Smoothness is estimated, not searched.**")
    long_list = text.index("\n## ", claims_start)
    assert first_fence < claims_start < long_list
    assert "Rdatasets" in text[first_fence:claims_start], "the first example must use real data"
    claims = re.findall(r"^- \*\*.*?(?=^- \*\*|^\S|\Z)", text[claims_start:long_list], re.MULTILINE | re.DOTALL)
    assert len(claims) == 3, claims
    for claim in claims:
        assert re.search(r"\]\((?:https?://|docs/)[^)]+\)", claim), f"claim without evidence link: {claim[:60]}"

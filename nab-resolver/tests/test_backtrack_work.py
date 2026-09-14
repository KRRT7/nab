"""Backtracking visits only the package trails whose assignments it removes."""

from __future__ import annotations

from collections import UserDict

import pytest

from nab_resolver.partial_solution import Assignment, PartialSolution
from nab_resolver.ranges import Range
from nab_resolver.types import Incompatibility, IncompatibilityCause


class CountingTrails(UserDict[str, list[Assignment[str, int]]]):
    """Count trail lookups, including lookups through mapping iteration."""

    def __init__(self, entries: dict[str, list[Assignment[str, int]]]) -> None:
        super().__init__(entries)
        self.visits: list[str] = []

    def __getitem__(self, package: str) -> list[Assignment[str, int]]:
        """Record a visit before returning the requested trail."""
        self.visits.append(package)
        return super().__getitem__(package)


@pytest.mark.parametrize("retained", [0, 32, 512])
@pytest.mark.parametrize("removed", [0, 1, 3])
def test_trail_visits_depend_on_removed_packages(
    monkeypatch: pytest.MonkeyPatch, retained: int, removed: int
) -> None:
    solution: PartialSolution[str, int] = PartialSolution()
    for index in range(retained):
        solution.decide(f"kept-{index}", 0)
    for index in range(removed):
        solution.decide(f"removed-{index}", 1)
    trails = CountingTrails(solution._assignments_by_package)
    monkeypatch.setattr(solution, "_assignments_by_package", trails)

    solution.backtrack(retained)

    assert sorted(trails.visits) == sorted(f"removed-{i}" for i in range(removed))
    assert dict(solution.decisions()) == {f"kept-{i}": 0 for i in range(retained)}
    assert solution.trail_length == retained
    assert solution.decision_level == retained


def test_interleaved_changes_restore_each_affected_package_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution: PartialSolution[str, int] = PartialSolution()
    cause: Incompatibility[str, int] = Incompatibility(
        [], cause=IncompatibilityCause.ROOT
    )
    allowed = Range.at_least(1)
    excluded = Range.singleton(9)
    solution.decide("kept", 0)
    solution.derive("mixed", allowed, positive=True, cause=cause)
    solution.derive("mixed", excluded, positive=False, cause=cause)
    solution.derive("negative", excluded, positive=False, cause=cause)
    before_decisions = solution.decisions()
    before_ranges = solution.positive_ranges()
    mixed_before = solution.get("mixed")
    negative_before = solution.get("negative")

    solution.decide("mixed", 2)
    solution.derive("negative", allowed, positive=True, cause=cause)
    solution.derive("mixed", Range.singleton(3), positive=False, cause=cause)
    solution.decide("removed", 5)
    solution.derive("mixed", Range.less_than(8), positive=True, cause=cause)
    solution.derive("negative", Range.singleton(7), positive=False, cause=cause)
    later_decisions = solution.decisions()
    trails = CountingTrails(solution._assignments_by_package)
    monkeypatch.setattr(solution, "_assignments_by_package", trails)

    solution.backtrack(1)

    assert sorted(trails.visits) == ["mixed", "negative", "removed"]
    assert dict(solution.decisions()) == dict(before_decisions) == {"kept": 0}
    assert dict(solution.positive_ranges()) == dict(before_ranges)
    assert dict(later_decisions) == {"kept": 0, "mixed": 2, "removed": 5}
    assert solution.get("mixed") == mixed_before
    assert solution.get("negative") == negative_before
    assert solution.get("removed") is None
    assert solution.undecided_packages() == {"mixed"}
    assert solution.trail_length == 4

"""Cached restart decisions must agree with the mutable statistics mapping."""

from __future__ import annotations

import copy
import pickle
from collections import defaultdict
from collections.abc import Callable, ValuesView
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_resolver._conflict_counts import ConflictCounts
from nab_resolver.conflict import maybe_restart
from nab_resolver.ranges import Range
from nab_resolver.resolver import Resolver, ResolverStats

from .test_resolver import DictProvider


def test_increments_and_lowered_maximum() -> None:
    counts: ConflictCounts[str] = ConflictCounts(int)
    assert counts.maximum == 0
    counts["a"] += 5
    counts["b"] += 2
    assert counts.maximum == 5
    counts["a"] = 1
    assert counts.maximum == 2
    counts["b"] += 3
    assert counts.maximum == 5
    counts["a"] = 5
    counts["b"] = -10
    assert counts.maximum == 5
    counts["a"] = -20
    assert counts.maximum == -10
    counts.clear()
    counts["negative"] = -3
    assert counts.maximum == -3
    assert counts["missing"] == 0
    assert counts.maximum == 0


def test_increment_after_invalidating_before_repair() -> None:
    counts = ConflictCounts[str](int, a=10, b=2, c=3)
    assert counts.maximum == 10
    counts["a"] = 1
    counts["b"] += 5
    assert counts.maximum == 7


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("a", 0),
        lambda c: c.__delitem__("a"),
        lambda c: c.clear(),
        lambda c: c.pop("a"),
        lambda c: c.pop("absent", 0),
        lambda c: c.popitem(),
        lambda c: c.setdefault("new", 10),
        lambda c: c.setdefault("a", 20),
        lambda c: c.update({"a": -1}),
        lambda c: c.update([("new", 10)]),
        lambda c: c.update(a=0, new=20),
        lambda c: c.__ior__({"a": 0}),
        lambda c: c.__init__(int, {"a": 0}),
    ],
)
def test_public_mutators(mutate: Callable[[ConflictCounts[str]], object]) -> None:
    counts = ConflictCounts[str](int, {"b": 2, "a": 5})
    assert counts.maximum == 5
    mutate(counts)
    assert counts.maximum == max(counts.values(), default=0)
    assert counts.maximum == max(counts.values(), default=0)


@pytest.mark.parametrize("method", ["update", "__ior__"])
def test_partial_update_failure_invalidates(method: str) -> None:
    counts = ConflictCounts[str](int, {"a": 1})
    assert counts.maximum == 1
    with pytest.raises(ValueError, match="length 1; 2 is required"):
        getattr(counts, method)([("b", 10), ("invalid",)])
    assert counts == {"a": 1, "b": 10}
    assert counts.maximum == 10


def test_partial_reinitialization_failure_invalidates() -> None:
    counts = ConflictCounts[str](int, a=1)
    assert counts.maximum == 1
    with pytest.raises(ValueError, match="length 1; 2 is required"):
        counts.__init__(int, [("b", 10), ("invalid",)])
    assert counts.maximum == max(counts.values()) == 10


@pytest.mark.parametrize("method", ["__delitem__", "pop"])
def test_missing_deletion_does_not_insert(method: str) -> None:
    counts = ConflictCounts[str](int)
    assert counts.maximum == 0
    with pytest.raises(KeyError):
        getattr(counts, method)("absent")
    assert counts == {}
    assert counts.maximum == 0
    with pytest.raises(KeyError):
        counts.popitem()


def test_setdefault_retains_defaultdict_semantics() -> None:
    counts = ConflictCounts[str](int)
    assert counts.setdefault("a") is None
    assert counts["a"] is None


def test_default_factory_changes() -> None:
    counts = ConflictCounts[str](int, a=1)
    assert counts.maximum == 1
    counts.default_factory = lambda: 10
    assert counts["b"] == 10
    assert counts.maximum == 10
    counts.default_factory = None
    with pytest.raises(KeyError):
        counts["absent"]
    assert counts.maximum == 10


@pytest.mark.parametrize(
    "clone",
    [
        copy.copy,
        copy.deepcopy,
        lambda c: c.copy(),
        lambda c: pickle.loads(pickle.dumps(c)),  # noqa: S301
        lambda c: c | {"new": 20},
        lambda c: {"new": 20} | c,
    ],
)
def test_copy_pickle_and_union(clone: Callable[[Any], Any]) -> None:
    counts = ConflictCounts[str](int, {"a": 2})
    assert counts.maximum == 2
    cloned = clone(counts)
    assert isinstance(cloned, ConflictCounts)
    assert cloned.default_factory is int
    assert cloned.maximum == max(cloned.values())
    cloned["b"] += 30
    assert cloned.maximum == 30
    assert counts.maximum == 2
    assert counts == {"a": 2}


@pytest.mark.property
@given(
    st.lists(
        st.tuples(
            st.integers(0, 7),
            st.sampled_from("abc"),
            st.integers(-20, 20),
            st.booleans(),
        ),
        max_size=100,
    )
)
def test_mutation_sequences(operations: list[tuple[int, str, int, bool]]) -> None:
    counts: ConflictCounts[str] = ConflictCounts(int)
    reference: defaultdict[str, int] = defaultdict(int)
    for operation, key, value, check in operations:
        for mapping in (counts, reference):
            if operation == 0:
                mapping[key] += value
            elif operation == 1:
                mapping[key] = value
            elif operation == 2:
                mapping.pop(key, None)
            elif operation == 3:
                mapping.clear()
            elif operation == 4:
                mapping.update({key: value})
            elif operation == 5:
                mapping |= {key: value}
            elif operation == 6:
                mapping.setdefault(key, value)
            elif mapping:
                mapping.popitem()
        assert counts == reference
        if check:
            assert counts.maximum == max(reference.values(), default=0)
    assert counts.maximum == max(reference.values(), default=0)


def test_restart_retains_counts_and_rechecks_edits() -> None:
    resolver: Resolver[str, int] = Resolver(DictProvider({}))
    counts = resolver.stats.package_conflict_counts
    counts["a"] = 5
    assert maybe_restart(resolver, 5, 3) == (10, 2, True)
    assert resolver.stats.package_conflict_counts is counts
    assert maybe_restart(resolver, 10, 2) == (10, 2, False)
    counts.update(a=10)
    assert maybe_restart(resolver, 10, 2) == (20, 1, True)
    counts["a"] = 100
    counts.clear()
    assert maybe_restart(resolver, 20, 1) == (20, 1, False)
    assert resolver.stats.restarts == 2


@pytest.mark.parametrize("replace_stats", [False, True])
def test_replaced_mapping_preserves_external_alias(replace_stats: bool) -> None:
    resolver: Resolver[str, int] = Resolver(DictProvider({}))
    supplied: defaultdict[str, int] = defaultdict(int, a=5)
    if replace_stats:
        resolver.stats = ResolverStats(package_conflict_counts=supplied)
    else:
        resolver.stats.package_conflict_counts = supplied
    assert resolver.stats.package_conflict_counts is supplied
    assert maybe_restart(resolver, 5, 3) == (10, 2, True)
    supplied["a"] = 10
    assert maybe_restart(resolver, 10, 2) == (20, 1, True)
    supplied.clear()
    assert maybe_restart(resolver, 20, 1) == (20, 1, False)


def test_conflicts_and_resolver_reuse() -> None:
    resolver = Resolver(
        DictProvider(
            {
                "root": {1: {"a": Range.full(), "b": Range.full()}},
                "a": {v: {"c": Range.singleton(v)} for v in range(20, 0, -1)},
                "b": {v: {"c": Range.less_than(3)} for v in range(20, 0, -1)},
                "c": {v: {} for v in range(20, 0, -1)},
                "easy": {1: {}},
            }
        )
    )
    assert resolver.resolve({"root": Range.singleton(1)})["c"] < 3
    counts = resolver.stats.package_conflict_counts
    assert isinstance(counts, ConflictCounts)
    assert counts.maximum == max(counts.values()) > 0
    assert resolver.resolve({"easy": Range.full()}) == {"easy": 1}
    fresh = resolver.stats.package_conflict_counts
    assert fresh is not counts
    assert isinstance(fresh, ConflictCounts)
    assert fresh.maximum == 0
    counts["old"] = 1000
    assert maybe_restart(resolver, 5, 3) == (5, 3, False)


def test_warm_restart_checks_do_not_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    scans = 0
    original = ConflictCounts.values

    def values(counts: ConflictCounts[str]) -> ValuesView[int]:
        nonlocal scans
        scans += 1
        return original(counts)

    monkeypatch.setattr(ConflictCounts, "values", values)
    resolver: Resolver[str, int] = Resolver(DictProvider({}))
    counts = resolver.stats.package_conflict_counts
    for i in range(512):
        counts[str(i)] += 1
    for _ in range(100):
        counts["0"] += 1
        assert maybe_restart(resolver, 1000, 3) == (1000, 3, False)
    assert scans == 0
    counts.update({"0": 1})
    assert maybe_restart(resolver, 1000, 0) == (1000, 0, False)
    assert scans == 0
    assert maybe_restart(resolver, 1000, 3) == (1000, 3, False)
    assert scans == 1
    assert maybe_restart(resolver, 1000, 3) == (1000, 3, False)
    assert scans == 1

"""Resolver-owned conflict counters with a lazily repaired maximum."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ._compat import override
from .types import PackageType


class ConflictCounts(defaultdict[PackageType, int]):
    """Maintain the maximum during increments; rescan after other mutations.

    Statistics remain mutable through the normal defaultdict interface. Bulk
    mutators bypass ``__setitem__`` in dict, so they explicitly invalidate the
    cache, including when an update fails after inserting some items.
    Explicit calls to dict's base-class mutators bypass these overrides and
    must not be used on this mapping.

    Only resolver-owned maps use this class. Caller-supplied maps retain their
    identity and use the uncached restart check instead.
    """

    __slots__ = ("_maximum",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Accept defaultdict's constructor forms, also used by copy and pickle.
        self._maximum: int | None = None
        super().__init__(*args, **kwargs)

    @property
    def maximum(self) -> int:
        """Return max(values(), default=0), repairing an invalidated cache once."""
        if self._maximum is None:
            self._maximum = max(self.values(), default=0)
        return self._maximum

    @override
    def __missing__(self, key: PackageType) -> int:
        """Route default-factory insertion through the tracked assignment."""
        factory = self.default_factory
        if factory is None:
            raise KeyError(key)
        value = factory()
        self[key] = value
        return value

    @override
    def __setitem__(self, key: PackageType, value: int) -> None:
        """Keep the cached maximum exact for ordinary counter increments."""
        previous = self.get(key)
        super().__setitem__(key, value)
        if len(self) == 1:
            self._maximum = value
        elif self._maximum is not None:
            if value >= self._maximum:
                self._maximum = value
            elif previous == self._maximum:
                self._maximum = None

    @override
    def __delitem__(self, key: PackageType) -> None:
        """Invalidate after a successful deletion."""
        super().__delitem__(key)
        self._maximum = None

    @override
    def clear(self) -> None:
        """Empty the counters and reset the maximum."""
        super().clear()
        self._maximum = 0

    @override
    def pop(self, *args: Any) -> Any:
        """Remove an entry and invalidate the maximum."""
        result = super().pop(*args)
        self._maximum = None
        return result

    @override
    def popitem(self) -> tuple[PackageType, int]:
        """Remove the last entry and invalidate the maximum."""
        result = super().popitem()
        self._maximum = None
        return result

    @override
    def setdefault(self, key: PackageType, default: Any = None, /) -> Any:
        """Insert a default and invalidate the maximum."""
        result = super().setdefault(key, default)
        self._maximum = None
        return result

    @override
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Invalidate even if a bulk update only partially succeeds."""
        try:
            super().update(*args, **kwargs)
        finally:
            self._maximum = None

    @override
    def __ior__(self, other: Any) -> Any:
        """Invalidate after an in-place union, including partial failures."""
        try:
            super().__ior__(other)
        finally:
            self._maximum = None
        return self

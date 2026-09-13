"""Compatibility reuse must preserve target separation and bounded storage."""

from collections.abc import Iterator

import pytest

from nab_provider import tags


@pytest.fixture(autouse=True)
def clean_axis_cache() -> Iterator[None]:
    tags._python_axis_accepts_tags.cache_clear()
    yield
    tags._python_axis_accepts_tags.cache_clear()


def test_distinct_packages_and_releases_share_compatibility_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = tags._python_axis_tags

    def counted(version: str, implementation: str) -> frozenset[tuple[str, str]]:
        nonlocal calls
        calls += 1
        return original(version, implementation)

    monkeypatch.setattr(tags, "_python_axis_tags", counted)
    for i in range(512):
        assert tags.python_axis_accepts(
            "3.11", "cpython", f"package{i}-{i}.0-cp311-cp311-manylinux_2_17_x86_64.whl"
        )
    assert calls == 1


def test_target_changes_do_not_reuse_another_axes_answer() -> None:
    wheels = {
        "pkg-1-cp310-cp310-any.whl": [True, False, False, False],
        "pkg-1-pp310-pypy310_pp73-any.whl": [False, True, False, False],
        "pkg-1-cp313-cp313t-any.whl": [False, False, True, False],
        "pkg-1-cp310-abi3-any.whl": [True, False, True, False],
        "pkg-1-py2.py3-none-any.whl": [True, True, True, True],
    }
    targets = [
        ("3.10", "cpython"),
        ("3.10", "pypy"),
        ("3.13", "cpython"),
        ("3.7", "cpython"),
    ]
    for _ in range(2):
        for wheel, expected in wheels.items():
            assert [
                tags.python_axis_accepts(*target, wheel) for target in targets
            ] == expected


def test_invalid_filename_and_unknown_implementation_bypass_cache() -> None:
    assert tags.python_axis_accepts("3.11", "cpython", "pkg-1.tar.gz")
    assert tags.python_axis_accepts("3.11", "cpython", "not-a-wheel.whl")
    assert tags.python_axis_accepts("3.11", "graalpy", "pkg-1-cp27-cp27m-any.whl")
    assert tags._python_axis_accepts_tags.cache_info().currsize == 0


def test_compressed_tag_order_does_not_change_the_answer() -> None:
    assert tags.python_axis_accepts("3.11", "cpython", "a-1-py2.py3-none-any.whl")
    assert tags.python_axis_accepts("3.11", "cpython", "b-2-py3.py2-none-any.whl")
    assert tags._python_axis_accepts_tags.cache_info().misses == 1
    assert tags._python_axis_accepts_tags.cache_info().hits == 1


def test_cache_evicts_and_recomputes_without_changing_the_answer() -> None:
    for i in range(8193):
        assert tags.python_axis_accepts(
            "3.11", "cpython", f"pkg-1-py3-none-platform{i}.whl"
        )
    before = tags._python_axis_accepts_tags.cache_info()
    assert before.currsize == before.maxsize == 8192
    assert tags.python_axis_accepts("3.11", "cpython", "pkg-1-py3-none-platform0.whl")
    after = tags._python_axis_accepts_tags.cache_info()
    assert after.currsize == 8192
    assert after.misses == before.misses + 1


@pytest.mark.parametrize(
    ("interpreters", "abis", "expected"),
    [
        ("cp38.cp39.cp310", "cp38.cp39.cp310", False),
        ("cp38.cp39.cp310", "cp38.cp39.abi3", True),
        ("py2.py3", "unknown.none", True),
        ("cp311", "unknown", False),
    ],
)
def test_compressed_pairs_preserve_acceptance(
    interpreters: str, abis: str, *, expected: bool
) -> None:
    filename = f"pkg-1-{interpreters}-{abis}-linux_x86_64.win_amd64.whl"
    assert tags.python_axis_accepts("3.11", "cpython", filename) is expected
    assert tags.python_axis_accepts("3.11", "cpython", filename) is expected


def test_each_compressed_pair_can_supply_the_match() -> None:
    filename = "pkg-1-cp38.cp39.cp310.cp311-cp38.cp39.cp310.cp311-any.whl"
    for version in ("3.8", "3.9", "3.10", "3.11"):
        assert tags.python_axis_accepts(version, "cpython", filename)

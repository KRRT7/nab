"""Resolve sparse interpreter targets and select their merged lock."""

from __future__ import annotations

import pytest

from nab_index.client import WheelFile
from nab_project._testing.coordinator_fake import FakeFetchPort, make_coordinator
from nab_project.inputs import ResolveInputs
from nab_project.lockfile import build_pylock
from nab_project.resolve import build_lock_input, resolve_with_coordinator
from nab_provider._vendor.packaging.pylock import PylockSelectError
from nab_provider._vendor.packaging.requirements import Requirement
from nab_provider._vendor.packaging.tags import Tag
from nab_provider.provider import BuildPolicy
from nab_provider.tags import PlatformSpec
from nab_provider.target import Matrix, ResolveTarget


def _wheel(package: str, version: str) -> WheelFile:
    """A universal wheel served by the in-memory index."""
    return WheelFile(
        filename=f"{package}-{version}-py3-none-any.whl",
        url=f"https://example.com/{package}-{version}-py3-none-any.whl",
        version=version,
        requires_python=None,
        has_metadata=True,
        upload_time=None,
        hashes=(("sha256", "a" * 64),),
    )


def _interpreter_graph() -> FakeFetchPort:
    """Serve divergent interpreter pins and independent platform dependencies."""
    listings = {
        "base": [_wheel("base", "1.0")],
        "variant": [_wheel("variant", "1.0"), _wheel("variant", "2.0")],
        "pypydep": [_wheel("pypydep", "1.0")],
        "windowsdep": [_wheel("windowsdep", "1.0")],
    }
    requires = {
        "base": (
            'variant==1.0 ; implementation_name == "cpython"',
            'variant==2.0 ; implementation_name == "pypy"',
            'pypydep ; implementation_name == "pypy"',
            'windowsdep ; sys_platform == "win32"',
        )
    }
    metadata = {}
    for name, wheels in listings.items():
        for wheel in wheels:
            assert wheel.metadata_url is not None
            dependencies = "".join(
                f"Requires-Dist: {dep}\n" for dep in requires.get(name, ())
            )
            metadata[wheel.metadata_url] = (
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {wheel.version}\n"
                f"{dependencies}\n"
            )
    return make_coordinator(listings=listings, metadata_by_url=metadata)


@pytest.mark.parametrize("cpython_range", [">=3.10,<3.16", ">=3.12,<3.16"])
def test_sparse_lock_selects_only_declared_interpreters(cpython_range: str) -> None:
    """Merged pins and support markers preserve the missing PyPy corners."""
    targets = Matrix(
        python=">=3.10,<3.16",
        platforms=(PlatformSpec("linux_x86_64"), PlatformSpec("windows_amd64")),
        implementations=("cpython", "pypy"),
        implementation_python={"cpython": cpython_range, "pypy": "==3.11.*"},
    ).expand()
    inputs = ResolveInputs(build_policy=BuildPolicy.NEVER)
    result = resolve_with_coordinator(
        _interpreter_graph(), targets, [Requirement("base")], inputs=inputs
    )

    assert result.success
    assert len(result.target_results) == len(targets)
    pylock = build_pylock(build_lock_input(result, inputs=inputs))
    pylock.validate()

    for target in targets:
        selected = {
            str(package.name): str(package.version)
            for package, _ in pylock.select(
                environment=target.marker_env, tags=[Tag("py3", "none", "any")]
            )
        }
        expected = {
            "base": "1.0",
            "variant": "2.0" if target.implementation == "pypy" else "1.0",
        }
        if target.implementation == "pypy":
            expected["pypydep"] = "1.0"
        assert target.platform_spec is not None
        if target.platform_spec.platform_id == "windows_amd64":
            expected["windowsdep"] = "1.0"
        assert selected == expected

    for version in ("3.10", "3.12", "3.15"):
        undeclared = ResolveTarget.for_declared(
            python_version=version,
            spec=PlatformSpec("linux_x86_64"),
            implementation="pypy",
        )
        with pytest.raises(PylockSelectError, match="environments specified"):
            list(
                pylock.select(
                    environment=undeclared.marker_env,
                    tags=[Tag("py3", "none", "any")],
                )
            )


def test_sparse_lock_restricts_implementations_with_identical_pins() -> None:
    """An implementation's Python range limits support even for identical pins."""
    targets = Matrix(
        python=">=3.11,<3.13",
        platforms=(PlatformSpec("linux_x86_64"),),
        implementations=("cpython", "pypy"),
        implementation_python={"cpython": "==3.12.*", "pypy": "==3.11.*"},
    ).expand()
    coordinator = make_coordinator(
        listings={"base": [_wheel("base", "1.0")]}, auto_metadata=True
    )
    inputs = ResolveInputs(build_policy=BuildPolicy.NEVER)
    result = resolve_with_coordinator(
        coordinator, targets, [Requirement("base")], inputs=inputs
    )
    assert result.success
    pylock = build_pylock(build_lock_input(result, inputs=inputs))
    pylock.validate()

    for target in targets:
        selected = list(
            pylock.select(
                environment=target.marker_env, tags=[Tag("py3", "none", "any")]
            )
        )
        assert [str(package.name) for package, _ in selected] == ["base"]

    for version, implementation in (("3.11", "cpython"), ("3.12", "pypy")):
        undeclared = ResolveTarget.for_declared(
            python_version=version,
            spec=PlatformSpec("linux_x86_64"),
            implementation=implementation,
        )
        with pytest.raises(PylockSelectError, match="environments specified"):
            list(
                pylock.select(
                    environment=undeclared.marker_env,
                    tags=[Tag("py3", "none", "any")],
                )
            )

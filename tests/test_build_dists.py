"""Tests for the distribution build helpers in tasks/build_dists.py."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tasks" / "build_dists.py"
_spec = importlib.util.spec_from_file_location("nab_build_dists", _PATH)
assert _spec is not None
assert _spec.loader is not None
build_dists = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dists)


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the published-file verifier with its real release helpers."""
    monkeypatch.syspath_prepend(str(_PATH.parent))
    monkeypatch.setitem(sys.modules, "build_dists", build_dists)
    path = _PATH.with_name("verify_published.py")
    spec = importlib.util.spec_from_file_location("nab_verify_published", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wheel_tree(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Write one metadata-bearing wheel for every release package."""
    wheels = []
    for package in build_dists.PACKAGES:
        stem = package.replace("-", "_")
        directory = tmp_path / package
        directory.mkdir()
        wheel = directory / f"{stem}-0.0.3-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"{stem}-0.0.3.dist-info/METADATA",
                "Name: " + package + "\nVersion: 0.0.3\n",
            )
        wheels.append(wheel)
    return tmp_path, wheels


def test_source_dir_maps_umbrella_to_root() -> None:
    assert build_dists.source_dir("nab") == build_dists.REPO_ROOT
    assert (
        build_dists.source_dir("nab-resolver") == build_dists.REPO_ROOT / "nab-resolver"
    )


def test_artifact_hashes_picks_only_distributions(tmp_path: Path) -> None:
    pkg = tmp_path / "nab"
    pkg.mkdir()
    (pkg / "nab-0.0.3-py3-none-any.whl").write_bytes(b"wheel")
    (pkg / "nab-0.0.3.tar.gz").write_bytes(b"sdist")
    (pkg / "notes.txt").write_bytes(b"ignore me")

    hashes = build_dists._artifact_hashes(tmp_path)

    assert set(hashes) == {"nab-0.0.3-py3-none-any.whl", "nab-0.0.3.tar.gz"}
    assert hashes["nab-0.0.3.tar.gz"] == hashlib.sha256(b"sdist").hexdigest()


def test_smoke_installs_every_built_wheel_and_checks_both_entries(
    wheel_tree: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    dist, wheels = wheel_tree
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> str:
        calls.append(command)
        return "nab 0.0.3\n" if "--version" in command else ""

    monkeypatch.setattr(build_dists, "_run", run)
    monkeypatch.setattr(
        build_dists.shutil, "which", lambda name, *, path: str(Path(path) / name)
    )
    build_dists.smoke_test(dist)

    installs = [
        command
        for command in calls
        if command[1:4] == ["-m", "pip", "install"] and "--upgrade" not in command
    ]
    assert len(installs) == 1
    assert installs[0][4:] == [str(wheel) for wheel in wheels]

    versions = [command for command in calls if command[-1] == "--version"]
    assert len(versions) == 2
    assert versions[0][1:] == ["-m", "nab", "--version"]
    assert Path(versions[1][0]).name == "nab"
    assert any(command[-1] == "--help" for command in calls)


@pytest.mark.parametrize("entry", ["module", "console"])
@pytest.mark.parametrize("reported", ["0.0.4", "0.0.30"])
def test_smoke_rejects_a_wrong_installed_version(
    wheel_tree: tuple[Path, list[Path]],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    reported: str,
) -> None:
    dist, _ = wheel_tree

    def run(command: list[str], **kwargs: object) -> str:
        selected = "module" if command[1:3] == ["-m", "nab"] else "console"
        return f"nab {reported if selected == entry else '0.0.3'}"

    monkeypatch.setattr(build_dists, "_run", run)
    monkeypatch.setattr(
        build_dists.shutil, "which", lambda name, *, path: str(Path(path) / name)
    )
    with pytest.raises(SystemExit, match="expected.*0.0.3"):
        build_dists.smoke_test(dist)


@pytest.mark.parametrize("copies", [0, 2])
def test_smoke_requires_one_wheel_for_every_package(
    wheel_tree: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch, copies: int
) -> None:
    dist, wheels = wheel_tree
    wheel = wheels[0]
    if copies == 0:
        wheel.unlink()
    else:
        wheel.with_name("other.whl").write_bytes(wheel.read_bytes())
    calls: list[list[str]] = []
    monkeypatch.setattr(
        build_dists, "_run", lambda command, **kwargs: calls.append(command) or ""
    )
    with pytest.raises(SystemExit, match="exactly one wheel"):
        build_dists.smoke_test(dist)
    assert calls == []


@pytest.mark.parametrize("copies", [0, 2])
def test_published_verification_requires_one_local_artifact(
    tmp_path: Path, published: ModuleType, monkeypatch: pytest.MonkeyPatch, copies: int
) -> None:
    filename = "nab-0.0.3.tar.gz"
    for index in range(copies):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / filename).write_bytes(b"built")

    calls: list[str] = []
    monkeypatch.setattr(
        published, "_verify_attestation", lambda *args: calls.append("attestation")
    )
    monkeypatch.setattr(
        published,
        "_published_sha256",
        lambda *args: hashlib.sha256(b"built").hexdigest(),
    )
    with pytest.raises(published.VerificationError, match="exactly one"):
        published._check_once("nab", "0.0.3", filename, "repository", tmp_path)
    assert calls == []


def test_release_checks_all_local_artifacts_before_remote_work(
    tmp_path: Path, published: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = published.expected_files("0.0.3")
    for _, filename in files[:-1]:
        (tmp_path / filename).write_bytes(b"built")
    calls: list[str] = []
    monkeypatch.setattr(
        published, "_verify_attestation", lambda *args: calls.append("attestation")
    )
    monkeypatch.setattr(
        published,
        "_published_sha256",
        lambda *args: hashlib.sha256(b"built").hexdigest(),
    )
    with pytest.raises(SystemExit, match="exactly one"):
        published.verify_release("0.0.3", tmp_path)
    assert calls == []


def test_published_verification_preserves_attestation_only_mode(
    published: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        published, "_verify_attestation", lambda *args: calls.append(args)
    )
    result = published._check_once(
        "nab", "0.0.3", "nab-0.0.3.tar.gz", "repository", None
    )
    assert result == "attestation OK"
    assert calls == [("nab-0.0.3.tar.gz", "repository")]


@pytest.mark.parametrize("matches", [False, True])
def test_published_verification_compares_the_built_bytes(
    tmp_path: Path,
    published: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    matches: bool,
) -> None:
    filename = "nab-0.0.3.tar.gz"
    (tmp_path / filename).write_bytes(b"built")
    monkeypatch.setattr(published, "_verify_attestation", lambda *args: None)
    payload = b"built" if matches else b"different"
    monkeypatch.setattr(
        published,
        "_published_sha256",
        lambda *args: hashlib.sha256(payload).hexdigest(),
    )
    if matches:
        assert (
            published._check_once("nab", "0.0.3", filename, "repository", tmp_path)
            == "attestation OK; published bytes match built bytes"
        )
    else:
        with pytest.raises(published.VerificationError, match="published bytes differ"):
            published._check_once("nab", "0.0.3", filename, "repository", tmp_path)

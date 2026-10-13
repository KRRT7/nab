"""Lock sparse interpreter matrices through the command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nab.cli import run
from nab_project.toml_io import loads
from nab_provider._vendor.packaging.pylock import Pylock, PylockSelectError
from nab_provider.tags import PlatformSpec
from nab_provider.target import ResolveTarget

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("declaration", ["pyproject", "nab.toml"])
@pytest.mark.parametrize("shared_python", [">=3.10,<3.16", "==3.12"])
def test_lock_declares_only_selected_interpreter_targets(
    tmp_path: Path,
    declaration: str,
    shared_python: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """File overrides survive CLI changes to the shared matrix defaults."""
    project = tmp_path / "pyproject.toml"
    body = (
        '[project]\nname = "example"\nversion = "1.0"\n'
        'requires-python = ">=3.10"\ndependencies = []\n'
    )
    root = "tool.nab" if declaration == "pyproject" else ""
    matrix = f"{root}.matrix" if root else "matrix"
    config = (
        (f"[{root}]\n" if root else "")
        + 'mode = "universal"\n'
        + f'[{matrix}]\npython = ">=3.10,<3.16"\n'
        + 'platforms = ["linux_x86_64", "macos_arm64"]\n'
        + 'implementations = ["cpython", "pypy"]\n'
        + f'[{matrix}.overrides.implementations.pypy]\npython = "==3.11"\n'
        + 'platforms = ["linux_x86_64"]\n'
    )
    if declaration == "pyproject":
        body += config
    else:
        (tmp_path / "nab.toml").write_text(config, encoding="utf-8")
    project.write_text(body, encoding="utf-8")
    flags = ["--project-matrix-python", shared_python]

    assert run(["config", "get", "matrix", "--path", str(project), *flags]) == 0
    rendered = capsys.readouterr().out
    assert "implementations=['cpython', 'pypy']" in rendered
    assert (
        "overrides={'implementations': {'pypy': {'python': '==3.11', 'platforms': ['linux_x86_64']}}}"
        in rendered
    )

    output = tmp_path / "pylock.toml"
    assert (
        run(
            [
                "lock",
                str(project),
                "--offline",
                "--no-cache",
                "--format",
                "pylock",
                "--output",
                str(output),
                *flags,
            ]
        )
        == 0
    )

    pylock = Pylock.from_dict(loads(output.read_text(encoding="utf-8")))
    pylock.validate()
    assert pylock.requires_python is not None
    assert str(pylock.requires_python) == ">=3.10"
    assert pylock.environments is not None
    assert len(pylock.environments) == (13 if shared_python == ">=3.10,<3.16" else 3)

    for minor in ("3.10", "3.11", "3.12", "3.13", "3.14", "3.15"):
        for platform in ("linux_x86_64", "macos_arm64", "windows_amd64"):
            for implementation in ("cpython", "pypy"):
                target = ResolveTarget.for_declared(
                    python_version=minor,
                    spec=PlatformSpec(platform),
                    implementation=implementation,
                )
                cpython_supported = (
                    implementation == "cpython"
                    and platform != "windows_amd64"
                    and (shared_python == ">=3.10,<3.16" or minor == "3.12")
                )
                pypy_supported = (
                    implementation == "pypy"
                    and minor == "3.11"
                    and platform == "linux_x86_64"
                )
                if cpython_supported or pypy_supported:
                    assert (
                        list(pylock.select(environment=target.marker_env, tags=[]))
                        == []
                    )
                else:
                    with pytest.raises(PylockSelectError):
                        list(pylock.select(environment=target.marker_env, tags=[]))


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", "overrides={'implementations': {'pypy': {}}}"),
        (
            'python = "==3.11"\n',
            "overrides={'implementations': {'pypy': {'python': '==3.11'}}}",
        ),
        (
            'platforms = ["windows_amd64"]\n',
            "overrides={'implementations': {'pypy': {'platforms': ['windows_amd64']}}}",
        ),
    ],
)
def test_config_inspection_shows_only_explicit_implementation_fields(
    tmp_path: Path, body: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inspection distinguishes inherited axes from explicitly replaced values."""
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "example"\nversion = "1.0"\ndependencies = []\n'
        '[tool.nab]\nmode = "universal"\n'
        '[tool.nab.matrix]\npython = "==3.12"\n'
        'platforms = ["linux_x86_64"]\nimplementations = ["cpython", "pypy"]\n'
        "[tool.nab.matrix.overrides.implementations.pypy]\n" + body,
        encoding="utf-8",
    )
    assert run(["config", "get", "matrix", "--path", str(project)]) == 0
    assert expected in capsys.readouterr().out

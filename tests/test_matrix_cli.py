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


@pytest.mark.parametrize("declaration", ["file", "cli"])
def test_lock_declares_only_selected_interpreter_minors(
    tmp_path: Path, declaration: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The project table and its CLI equivalent produce the same support set."""
    project = tmp_path / "pyproject.toml"
    body = (
        '[project]\nname = "example"\nversion = "1.0"\n'
        'requires-python = ">=3.10"\ndependencies = []\n'
    )
    flags: list[str] = []
    if declaration == "file":
        body += (
            '[tool.nab]\nmode = "universal"\n'
            '[tool.nab.matrix]\npython = ">=3.10,<3.16"\n'
            'platforms = ["linux_x86_64", "macos_arm64"]\n'
            'implementations = ["cpython", "pypy"]\n'
            '[tool.nab.matrix.implementation-python]\npypy = "==3.11"\n'
        )
    else:
        flags = [
            "--project-mode",
            "universal",
            "--project-matrix-python",
            ">=3.10,<3.16",
            "--project-matrix-platforms",
            "linux_x86_64",
            "macos_arm64",
            "--project-matrix-implementations",
            "cpython",
            "pypy",
            "--project-matrix-implementation-python",
            "pypy===3.11",
        ]
    project.write_text(body, encoding="utf-8")
    output = tmp_path / "pylock.toml"

    assert run(["config", "get", "matrix", "--path", str(project), *flags]) == 0
    rendered = capsys.readouterr().out
    assert "implementations=['cpython', 'pypy']" in rendered
    assert "implementation-python={'pypy': '==3.11'}" in rendered

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
    assert len(pylock.environments) == 14

    for minor in ("3.10", "3.11", "3.12", "3.13", "3.14", "3.15"):
        for platform in ("linux_x86_64", "macos_arm64"):
            for implementation in ("cpython", "pypy"):
                target = ResolveTarget.for_declared(
                    python_version=minor,
                    spec=PlatformSpec(platform),
                    implementation=implementation,
                )
                if implementation == "cpython" or minor == "3.11":
                    assert (
                        list(pylock.select(environment=target.marker_env, tags=[]))
                        == []
                    )
                else:
                    with pytest.raises(PylockSelectError):
                        list(pylock.select(environment=target.marker_env, tags=[]))

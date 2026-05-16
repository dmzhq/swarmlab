"""Smoke tests — the toolchain is wired and the package imports."""

from __future__ import annotations

import swarmlab


def test_version_is_defined() -> None:
    assert isinstance(swarmlab.__version__, str)
    assert swarmlab.__version__.count(".") == 2


def test_version_is_semver_prefix() -> None:
    parts = swarmlab.__version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()

"""Smoke tests for demo scripts.

Each demo is executed as a subprocess. The test asserts the demo
completes with exit code 0. stderr is captured and surfaced on
failure so the first stack trace is visible in CI logs.

Demos are proper scripts with side effects (they print and solve on
import), so we use subprocess isolation rather than importing them
into the test process.

Slow demos are tagged with ``@pytest.mark.slow`` and skipped by
default. Run them explicitly with ``pytest -m slow``.
"""

import subprocess
import sys
from pathlib import Path

import pytest

DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"


def _run_demo(name: str) -> None:
    demo_path = DEMOS_DIR / name
    assert demo_path.exists(), f"Demo not found: {demo_path}"
    result = subprocess.run(
        [sys.executable, str(demo_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        msg = (
            f"Demo {name!r} exited with code {result.returncode}.\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
        pytest.fail(msg)


class TestDemoSmoke:
    """Each fast demo runs end-to-end and exits 0."""

    def test_lcp_as_mcp(self):
        _run_demo("lcp_as_mcp.py")

    def test_differentiable_lcp(self):
        _run_demo("differentiable_lcp.py")

    def test_nonlinear_1d_mcp(self):
        _run_demo("nonlinear_1d_mcp.py")

    def test_2d_nonlinear_complementarity_problem(self):
        _run_demo("2d_nonlinear_complementarity_problem.py")

    def test_kkt_conditions(self):
        _run_demo("kkt_conditions.py")

    def test_obstacle_problem(self):
        _run_demo("obstacle_problem.py")

    def test_spatial_price_equilibrium(self):
        _run_demo("spatial_price_equilibrium.py")

    def test_traffic_route_choice(self):
        _run_demo("traffic_route_choice.py")


class TestSlowDemoSmoke:
    """Demos that take several minutes — run with ``pytest -m slow``."""

    @pytest.mark.slow
    def test_bound_optimization(self):
        _run_demo("bound_optimization.py")

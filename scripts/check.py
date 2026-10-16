"""Run the pull-request checks in order, stopping at the first failure.

Usage: ``uv run python scripts/check.py [--fix] [step ...]``. Steps default to all of
lint, format, types, and tests. CI runs the same steps, so a local pass matches CI.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

TEST_ARGS = ("-m", "not installed_wheel and not live", "--cov=tbcore", "--cov-report=term-missing")


def commands(fix: bool) -> dict[str, list[str]]:
    # Tools run under this interpreter, so the script works in any environment
    # that has the dev extra installed (uv run, a pip venv, or CI).
    return {
        "lint": ["ruff", "check", ".", *(["--fix"] if fix else [])],
        "format": ["ruff", "format", "."] if fix else ["ruff", "format", "--check", "."],
        "types": ["mypy"],
        "tests": ["pytest", *TEST_ARGS],
    }


def main() -> int:
    steps = list(commands(fix=False))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", metavar="step", help=f"any of {', '.join(steps)}")
    parser.add_argument("--fix", action="store_true", help="apply ruff fixes and formatting")
    args = parser.parse_args()
    unknown = [step for step in args.steps if step not in steps]
    if unknown:
        parser.error(f"unknown step(s) {', '.join(unknown)}; choose from {', '.join(steps)}")

    table = commands(args.fix)
    for step in args.steps or steps:
        argv = [sys.executable, "-m", *table[step]]
        print(f"==> {step}: {shlex.join(table[step])}", flush=True)
        code = subprocess.run(argv, check=False).returncode
        if code != 0:
            print(f"==> {step} failed (exit {code})", file=sys.stderr)
            return code
    print("==> all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
